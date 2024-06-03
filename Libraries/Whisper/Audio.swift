//
//  Audio.swift
//  Whisper
//
//  Created by Derek Lewis on 5/29/24.
//

import AVFoundation
import MLX

public class AudioLoader {
    public static func loadAudio(file: String, sampleRate: Double) -> MLXArray? {
        let url = URL(fileURLWithPath: file)
        let fileFormat: AVAudioFormat
        let inputBuffer: AVAudioPCMBuffer
        
        do {
            let file = try AVAudioFile(forReading: url)
            fileFormat = file.processingFormat
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: fileFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
                return nil
            }
            
            try file.read(into: buffer)
            inputBuffer = buffer
        } catch {
            print("Failed to load audio: \(error)")
            return nil
        }
        
        // Define the desired output format
        let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        
        // Create an AVAudioConverter
        guard let converter = AVAudioConverter(from: fileFormat, to: outputFormat) else {
            print("Failed to create audio converter.")
            return nil
        }
        
        // Create the output buffer
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: AVAudioFrameCount(inputBuffer.frameLength)) else {
            return nil
        }
        
        // Perform the conversion
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }
        
        var error: NSError?
        let result = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
        if result != .haveData || error != nil {
            print("Failed to convert audio: \(String(describing: error))")
            return nil
        }
        
        guard let channelData = outputBuffer.floatChannelData else {
            return nil
        }
        
        let frameLength = Int(outputBuffer.frameLength)
        var audioData = [Float32](repeating: 0, count: frameLength)
        
        for i in 0..<frameLength {
            audioData[i] = channelData[0][i]
        }
        
        return MLXArray(audioData)
    }
}
