//
//  WhisperEvalApp.swift
//  WhisperEval
//
//  Created by Derek Lewis on 5/29/24.
//

import SwiftUI
import Whisper
import MLXRandom
import MLX

@main
struct WhisperEvalApp: App {
    
    init() {
//        MLXRandom.seed(123)
//        let batch = MLXRandom.normal([1, 102, 1280])
//        print("batch:", batch)
//        MLXRandom.seed(123)
//        let resid = Whisper.ResidualAttentionBlock(n_state: 1280, n_head: 20, cross_attention: false)
//        print(resid)
//        print("resid(batch):", resid(batch))
//        if let audioData = AudioLoader.loadAudio(file: "/Users/dlewis/sample.wav", sampleRate: 16000) {
//            print("Loaded audio data with \(audioData.count) samples.")
//            print(audioData)

        if let url = Bundle.main.url(forResource: "model_config", withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            let decoder = JSONDecoder()
            do {
                let whisperConfig = try decoder.decode(WhisperConfiguration.self, from: data)
                MLXRandom.seed(123)
                let whisperModel = Whisper(dims: whisperConfig)
                eval(whisperModel)
                MLXRandom.seed(123)
                let audio_input_tensor = MLXRandom.normal([1, whisperConfig.audioContextSize, whisperConfig.audioStateSize])
                eval(audio_input_tensor)
                print("audio_input_tensor:", audio_input_tensor)
                MLXRandom.seed(123)
                let text_input_tensor = MLXRandom.randInt(0..<whisperConfig.vocabularySize, [1, 1])
                print("text_input_tensor:", text_input_tensor)
        
                // Run the encoder
                let (embedding, kvCache, cross_qk) = whisperModel.decoder(text_input_tensor, xa: audio_input_tensor)
        
                print("embedding:", embedding)
                print("kvCache:", kvCache)
                print("cross_qk", cross_qk)
            } catch {
                print("Failed to decode JSON: \(error)")
            }
        } else {
            print("Failed to locate whisper_config.json")
        }
        
        // Run the encoder
//        let output_tensor = encoder(input_tensor)

        // test textdecoder
        
//        let n_mels = 80
//        let n_audio_ctx = 1500
//        let n_audio_state = 384
//        let n_audio_head = 6
//        let n_audio_layer = 4
//        let n_text_ctx=448
//        let n_text_state=384
//        let n_text_head=6
//        let n_text_layer=4
//        let n_vocab = 51865
//
//        MLXRandom.seed(123)
//        let decoder = TextDecoder(n_vocab: n_vocab, n_ctx: n_text_ctx, n_state: n_text_state, n_head: n_text_head, n_layer: n_text_layer)
//
//        // Create a random input tensor
//        MLXRandom.seed(123)
//        let audio_input_tensor = MLXRandom.normal([1, n_audio_ctx, n_audio_state])
//        print("audio_input_tensor:", audio_input_tensor)
//        MLXRandom.seed(123)
//        let text_input_tensor = MLXRandom.randInt(0..<n_vocab, [1, 1])
//        print("text_input_tensor:", text_input_tensor)
//
//        // Run the encoder
//        let (embedding, kvCache, cross_qk) = decoder(text_input_tensor, xa: audio_input_tensor)
//
//        print("embedding:", embedding)
//        print("kvCache:", kvCache)
//        print("cross_qk", cross_qk)
        
//        // Print the output
//        print("Output tensor shape: \(output_tensor.shape)")
//        print("Output tensor: \(output_tensor)")
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
