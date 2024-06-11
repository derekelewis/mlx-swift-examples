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
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .task {
                    await loadModelOnStartup()
                }
        }
    }
    
    @MainActor
    func loadModelOnStartup() async {
        do {
            let model = try await load_model(model_id: "mlx-community/whisper-tiny")
            print("Model loaded successfully: \(model)")
            let config = model.getDims()
            MLXRandom.seed(123)
            let audio_input_tensor = MLXRandom.normal([1, 2*config.audioContextSize, config.melsSize])
            print("audio_input_tensor:", audio_input_tensor)
            MLXRandom.seed(123)
            let text_input_tensor = MLXRandom.randInt(0..<config.vocabularySize, [1, 1])
            print("text_input_tensor:", text_input_tensor)
            let encoder_output = model.encoder(audio_input_tensor)
            let (logits, _, _) = model.decoder(text_input_tensor, xa: encoder_output)
            let tokenizer = try await WhisperTokenizer()
            print(logits)
            print(hanningWindow(length: 10))
        } catch {
            print("Failed to load model \(error)")
        }
    }
}
