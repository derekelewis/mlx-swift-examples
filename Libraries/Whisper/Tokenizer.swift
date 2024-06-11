//
//  Tokenizer.swift
//  Whisper
//
//  Created by Derek Lewis on 6/8/24.
//

import Foundation
import Tokenizers

public class WhisperTokenizer {
    let tokenizer: Tokenizer
    
    public init() async throws {
        self.tokenizer = try await load_tokenizer()
    }
}

public func load_tokenizer() async throws -> Tokenizer {
    let tokenizer = try await AutoTokenizer.from(pretrained: "openai/whisper-tiny")
    return tokenizer
}
