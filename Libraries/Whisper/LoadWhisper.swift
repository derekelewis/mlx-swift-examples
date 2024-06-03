//
//  LoadWhisper.swift
//  Whisper
//
//  Created by Derek Lewis on 6/3/24.
//

import Foundation
import Hub
import MLX
import MLXNN

public func load_model(model_id: String) async throws -> WhisperModel {
    do {
        let hub = HubApi()
        let repo = Hub.Repo(id: model_id)
        let modelFiles = ["*.npz"]
        let modelDirectory = try await hub.snapshot(from: repo, matching: modelFiles)
        
        // Process .npy files and add weights to a dictionary
        var weights = [String: MLXArray]()
        let npyFiles = try FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
        
        for npyFile in npyFiles where npyFile.pathExtension == "npy" {
            let weightArray = try MLX.loadArray(url: npyFile)
            let key = npyFile.deletingPathExtension().lastPathComponent
            weights[key] = weightArray
        }
        
        let config = try loadModelConfiguration()
        let model = WhisperModel(dims: config)
        
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        
        eval(model)
        
        return model
    } catch {
        throw NSError(domain: "LoadModelError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load model: \(error)"])
    }
}

public func loadModelConfiguration() throws -> WhisperConfiguration {
    guard let url = Bundle.main.url(forResource: "model_config", withExtension: "json") else {
        throw NSError(domain: "LoadModelError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to locate model_config.json"])
    }
    
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    
    do {
        let config = try decoder.decode(WhisperConfiguration.self, from: data)
        return config
    } catch {
        throw NSError(domain: "LoadModelError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to decode JSON: \(error)"])
    }
}
