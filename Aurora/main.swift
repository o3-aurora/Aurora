//
//  main.swift
//  Aurora
//
//  Created by Paxton Rivera on 6/3/25.
//

import Foundation

let command = "source ~/.zshrc && /usr/bin/python3 /Users/paxtonrivera/Documents/Aurora/advance.py"

// Launch the shell process
let process = Process()
process.executableURL = URL(fileURLWithPath: "/bin/zsh")
process.arguments = ["-c", command]

let pipe = Pipe()
process.standardOutput = pipe
process.standardError = pipe

do {
    try process.run()
} catch {
    print("Failed to run process: \(error)")
    exit(1)
}

// Read output in real-time
let handle = pipe.fileHandleForReading
handle.readabilityHandler = { fileHandle in
    if let line = String(data: fileHandle.availableData, encoding: .utf8) {
        print(line, terminator: "")
    }
}

process.waitUntilExit()
handle.readabilityHandler = nil

let status = process.terminationStatus
exit(Int32(status))
