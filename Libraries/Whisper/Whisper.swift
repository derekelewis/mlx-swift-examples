//
//  Whisper.swift
//  Whisper
//
//  Created by Derek Lewis on 5/29/24.
//

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXRandom

public typealias KVCacheElement = (MLXArray, MLXArray)

public func sinusoids(length: Int, channels: Int, max_timescale: Double = 10000.0) -> MLXArray {
    assert(channels % 2 == 0)
    let log_timescale_increment: Double = log(Double(max_timescale)) / Double(channels / 2 - 1)
    var a = MLXArray(0..<channels / 2).asType(.float32)
    a = -log_timescale_increment * a
    let inv_timescales = MLX.exp(a)
    let scaled_time: MLXArray = MLXArray(0..<length)[0..., .newAxis] * inv_timescales[.newAxis, 0...]
    return concatenated([sin(scaled_time), cos(scaled_time)], axis: 1)
}

public class MultiHeadAttention: Module {
    
    let n_state: Int
    let n_head: Int
    
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear
    
    public init(n_state: Int, n_head: Int) {
        
        self.n_state = n_state
        self.n_head = n_head
        self._query.wrappedValue = Linear(n_state, n_state)
        self._key.wrappedValue = Linear(n_state, n_state, bias: false)
        self._value.wrappedValue = Linear(n_state, n_state)
        self._out.wrappedValue = Linear(n_state, n_state)
    }
    
    public func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil, kvCache: KVCacheElement? = nil ) -> (MLXArray, KVCacheElement, MLXArray) {
        var q = query(x)
        var k = MLXArray()
        var v = MLXArray()
        
        if xa == nil {
            k = key(x)
            v = value(x)
            if let cache = kvCache {
                k = concatenated([cache.0, k], axis: 1)
                v = concatenated([cache.1, v], axis: 1)
            }
        } else if kvCache == nil {
            k = key(xa!)
            v = value(xa!)
        } else {
            k = kvCache!.0
            v = kvCache!.1
        }
        
        let (wv, qk) = self.qkvAttention(q: q, k: k, v: v, mask: mask)
        return (self.out(wv), (k, v), qk)
    }
    
    public func qkvAttention(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let (n_batch, n_ctx, n_state) = (q.dim(0), q.dim(1), q.dim(2))
        let base = self.n_state / self.n_head
        let scale = pow(Float(base), -0.25)
        
        let queries = q.reshaped(q.dim(0), q.dim(1), self.n_head, -1).transposed(0, 2, 1, 3) * scale
        let keys = k.reshaped(k.dim(0), k.dim(1), self.n_head, -1).transposed(0, 2, 3, 1) * scale
        let values = v.reshaped(v.dim(0), v.dim(1), self.n_head, -1).transposed(0, 2, 1, 3)
        
        var qk = MLX.matmul(queries, keys)
        if let mask = mask {
            qk = qk + mask[..<n_ctx, ..<n_ctx]
        }
        qk = qk.asType(.float32)
        
        let w = MLX.softmax(qk, axis: -1).asType(.float32)
        var out = MLX.matmul(w, values).transposed(0, 2, 1, 3)
        out = out.reshaped(n_batch, n_ctx, n_state)
        return (out, qk)
    }
}

public class ResidualAttentionBlock: Module {
    
    let n_state: Int
    let n_head: Int
    let cross_attention: Bool
    
    let attn: MultiHeadAttention
    let attn_ln: LayerNorm
    let cross_attn: MultiHeadAttention?
    let cross_attn_ln: LayerNorm?
    let mlp1: Linear
    let mlp2: Linear
    let mlp_ln: LayerNorm
    
    public init(n_state: Int, n_head: Int, cross_attention: Bool = false) {
        self.n_state = n_state
        self.n_head = n_head
        self.cross_attention = cross_attention
        
        self.attn = MultiHeadAttention(n_state: n_state, n_head: n_head)
        self.attn_ln = LayerNorm(dimensions: n_state)
        if cross_attention {
            self.cross_attn = MultiHeadAttention(n_state: n_state, n_head: n_head)
            self.cross_attn_ln = LayerNorm(dimensions: n_state)
        } else {
            self.cross_attn = nil
            self.cross_attn_ln = nil
        }
        
        let n_mlp = n_state * 4
        self.mlp1 = Linear(n_state, n_mlp)
        self.mlp2 = Linear(n_mlp, n_state)
        self.mlp_ln = LayerNorm(dimensions: n_state)
    }
    
    public func callAsFunction(_ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil, kvCache: (KVCacheElement?, KVCacheElement?)? = nil) -> (MLXArray, (KVCacheElement?, KVCacheElement?), MLXArray?) {
        var (kv, cross_kv) = kvCache ?? (nil, nil)
        var x_new = x

        let (y, new_kv, _) = self.attn(self.attn_ln(x), mask: mask, kvCache: kv)
        x_new = x_new + y
        var cross_qk: MLXArray? = nil

        if self.cross_attention, let cross_attn = self.cross_attn, let cross_attn_ln = self.cross_attn_ln {
            let (cross_y, new_cross_kv, new_cross_qk) = cross_attn(cross_attn_ln(x_new), xa: xa, kvCache: cross_kv)
            cross_kv = new_cross_kv
            cross_qk = new_cross_qk
            x_new = x_new + cross_y
        }
        x_new = x_new + self.mlp2(gelu(self.mlp1(self.mlp_ln(x_new))))
        return (x_new, (new_kv, cross_kv), cross_qk)
    }
}

public class AudioEncoder: Module {
    
    let n_mels: Int
    let n_ctx: Int
    let n_state: Int
    let n_head: Int
    let n_layer: Int
    
    let conv1: Conv1d
    let conv2: Conv1d
    let _positional_embedding: MLXArray
    let blocks: [ResidualAttentionBlock]
    let ln_post: LayerNorm
    
    public init(n_mels: Int, n_ctx: Int, n_state: Int, n_head: Int, n_layer: Int, dtype: MLX.DType = .float16) {
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        
        self.conv1 = Conv1d(inputChannels: n_mels, outputChannels: n_state, kernelSize: 3, padding: 1)
        self.conv2 = Conv1d(inputChannels: n_state, outputChannels: n_state, kernelSize: 3, stride: 2, padding: 1)
        self._positional_embedding = sinusoids(length: n_ctx, channels: n_state).asType(dtype)
        self.blocks = (0..<n_layer).map { _ in ResidualAttentionBlock(n_state: n_state, n_head: n_head) }
        self.ln_post = LayerNorm(dimensions: n_state)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var new_x: MLXArray = x
        new_x = gelu(self.conv1(new_x))
        new_x = gelu(self.conv2(new_x))
        let new_x_shape = ArraySlice(new_x.shape[1...])
        let positional_embedding_shape = ArraySlice(self._positional_embedding.shape)
        assert(new_x_shape == positional_embedding_shape, "incorrect audio shape")
        new_x = new_x + self._positional_embedding
        
        for block in self.blocks {
            (new_x, _, _) = block(new_x)
        }
        
        new_x = self.ln_post(new_x)
        return new_x
    }
}

public class TextDecoder: Module {
    
    let n_vocab: Int
    let n_ctx: Int
    let n_state: Int
    let n_head: Int
    let n_layer: Int
    
    let token_embedding: Embedding
    let positional_embedding: MLXArray
    let blocks: [ResidualAttentionBlock]
    let ln: LayerNorm
    let _mask: MLXArray
    
    public init(n_vocab: Int, n_ctx: Int, n_state: Int, n_head: Int, n_layer: Int, dtype: MLX.DType = .float16) {
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        
        self.token_embedding = Embedding(embeddingCount: n_vocab, dimensions: n_state)
        self.positional_embedding = zeros([n_ctx, n_state])
        
        self.blocks = (0..<n_layer).map { _ in ResidualAttentionBlock(n_state: n_state, n_head: n_head, cross_attention: true) }
        self.ln = LayerNorm(dimensions: n_state)
        self._mask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(n_ctx).asType(dtype)
    }
    
    public func callAsFunction(_ x: MLXArray, xa: MLXArray, kvCache: [(KVCacheElement?, KVCacheElement?)]? = nil) -> (MLXArray, [(KVCacheElement?, KVCacheElement?)], [MLXArray?]) {
        
        var offset: Int = 0
        if let kvCache = kvCache, let firstCache = kvCache.first {
            if let first = firstCache.0 {
                offset = first.0.shape[1]
            }
        }
        
        var new_x = (self.token_embedding(x) + self.positional_embedding[offset..<(offset + x.dim(-1))])
        
        var newKvCache: [(KVCacheElement?, KVCacheElement?)]
        var cross_qk: [MLXArray?] = []
        
        if kvCache == nil {
            newKvCache = [(KVCacheElement?, KVCacheElement?)](repeating: (nil, nil), count: self.blocks.count)
        } else {
            newKvCache = kvCache!
        }
        
        cross_qk = [MLXArray?](repeating: nil, count: self.blocks.count)
        
        for (e, block) in self.blocks.enumerated() {
            let (new_x_updated, new_kv, new_cross_qk) = block(new_x, xa: xa, mask: self._mask, kvCache: newKvCache[e])
            new_x = new_x_updated
            newKvCache[e] = new_kv
            cross_qk[e] = new_cross_qk
        }
        
        new_x = self.ln(new_x)
        return (matmul(new_x, self.token_embedding.weight.T), newKvCache, cross_qk)
    }
}

public class Whisper: Module {
    
    let dims: WhisperConfiguration
    
    public let encoder: AudioEncoder
    public let decoder: TextDecoder
    
    public init(dims: WhisperConfiguration, dtype: MLX.DType = .float32) {
        self.dims = dims
        MLXRandom.seed(123)
        self.encoder = AudioEncoder(n_mels: dims.melsSize, n_ctx: dims.audioContextSize, n_state: dims.audioStateSize, n_head: dims.audioAttentionHeads, n_layer: dims.audioLayers, dtype: dtype)
        MLXRandom.seed(123)
        self.decoder = TextDecoder(n_vocab: dims.vocabularySize, n_ctx: dims.textContextSize, n_state: dims.audioStateSize, n_head: dims.textAttentionHeads, n_layer: dims.textLayers, dtype: dtype)
    }
    
    public func embed_audio(mel: MLXArray) -> MLXArray {
        return self.encoder(mel)
    }
    
    public func logits(tokens: MLXArray, audio_features: MLXArray) -> MLXArray {
        return self.decoder(tokens, xa: audio_features).0
    }
    
    public func forward_with_cross_qk(mel: MLXArray, tokens: MLXArray) -> (MLXArray, [MLXArray?]) {
        let (logits, _, cross_qk) = self.decoder(tokens, xa: self.encoder(mel))
        return (logits, cross_qk)
    }
    
    public func call(mel: MLXArray, tokens: MLXArray) -> MLXArray {
        return self.decoder(tokens, xa: self.encoder(mel)).0
    }
    
    public func is_multilingual() -> Bool {
        return self.dims.vocabularySize >= 51865
    }
    
    public func num_languages() -> Int {
        return self.dims.vocabularySize - 51765 - (self.is_multilingual() ? 1 : 0)
    }
}

public struct WhisperConfiguration: Codable {
    public var melsSize: Int
    public var audioContextSize: Int
    public var audioStateSize: Int
    public var audioAttentionHeads: Int
    public var audioLayers: Int
    public var vocabularySize: Int
    public var textContextSize: Int
    public var textStateSize: Int
    public var textAttentionHeads: Int
    public var textLayers: Int
    
    enum CodingKeys: String, CodingKey {
        case melsSize = "n_mels"
        case audioContextSize = "n_audio_ctx"
        case audioStateSize = "n_audio_state"
        case audioAttentionHeads = "n_audio_head"
        case audioLayers = "n_audio_layer"
        case vocabularySize = "n_vocab"
        case textContextSize = "n_text_ctx"
        case textStateSize = "n_text_state"
        case textAttentionHeads = "n_text_head"
        case textLayers = "n_text_layer"
    }
    
    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container:
        KeyedDecodingContainer<WhisperConfiguration.CodingKeys> = try decoder.container(keyedBy: WhisperConfiguration.CodingKeys.self)
        
        self.melsSize = try container.decode(Int.self, forKey:  WhisperConfiguration.CodingKeys.melsSize)
        self.audioContextSize = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.audioContextSize)
        self.audioStateSize = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.audioStateSize)
        self.audioAttentionHeads = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.audioAttentionHeads)
        self.audioLayers = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.audioLayers)
        self.vocabularySize = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.vocabularySize)
        self.textContextSize = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.textContextSize)
        self.textStateSize = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.textStateSize)
        self.textAttentionHeads = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.textAttentionHeads)
        self.textLayers = try container.decode(Int.self, forKey: WhisperConfiguration.CodingKeys.textLayers)
    }
}
