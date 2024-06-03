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
            print("Failed to locate model_config.json")
        }
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
