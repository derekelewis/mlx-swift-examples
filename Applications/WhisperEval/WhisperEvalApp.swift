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
            var model = try await load_model(model_id: "mlx-community/whisper-tiny")
            print("Model loaded successfully: \(model)")
        } catch {
            print("Failed to load model \(error)")
        }
    }
}
