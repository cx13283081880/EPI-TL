//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this
//    list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Simple example for BFVrns (integer arithmetic)
 */

#include "openfhe.h"
#include <chrono>
#include <random> // 引入随机数生成库

using namespace lbcrypto;

int main() {
    // Sample Program: Step 1: Set CryptoContext
    auto start_total = std::chrono::high_resolution_clock::now();

    CCParams<CryptoContextBFVRNS> parameters;
    parameters.SetPlaintextModulus(65537);
    parameters.SetMultiplicativeDepth(3);

    CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);
    // Enable features that you wish to use
    cryptoContext->Enable(PKE);
    cryptoContext->Enable(KEYSWITCH);
    cryptoContext->Enable(LEVELEDSHE);

    // Sample Program: Step 2: Key Generation

    // Initialize Public Key Containers
    KeyPair<DCRTPoly> keyPair;

    // Generate a public/private key pair
    keyPair = cryptoContext->KeyGen();

    // Generate the relinearization key
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);

    // Sample Program: Step 3: Encryption

    // 生成 4097 个随机数的向量
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(1, 200); // 随机数范围在 1 到 100 之间
    int n=100;
    std::vector<int64_t> vectorOfInts1(n);
    std::vector<int64_t> vectorOfInts2(n);
    std::cout << "Maximum number of plaintext slots: " << cryptoContext->GetRingDimension() / 2 << std::endl;


    for (int i = 0; i < n; ++i) {
        vectorOfInts1[i] = dist(gen);
        vectorOfInts2[i] = dist(gen);
    }

    Plaintext plaintext1 = cryptoContext->MakePackedPlaintext(vectorOfInts1);
    Plaintext plaintext2 = cryptoContext->MakePackedPlaintext(vectorOfInts2);

    auto start_encrypt = std::chrono::high_resolution_clock::now();

    // The encoded vectors are encrypted
    auto ciphertext1 = cryptoContext->Encrypt(keyPair.publicKey, plaintext1);
    auto ciphertext2 = cryptoContext->Encrypt(keyPair.publicKey, plaintext2);

    auto end_encrypt = std::chrono::high_resolution_clock::now();

    // Sample Program: Step 4: Evaluation

    // Homomorphic additions
    auto start_add = std::chrono::high_resolution_clock::now();
    auto ciphertextAddResult = cryptoContext->EvalAdd(ciphertext1, ciphertext2);
    auto end_add = std::chrono::high_resolution_clock::now();
    
    // Homomorphic multiplications
    auto start_mult = std::chrono::high_resolution_clock::now();
    auto ciphertextMultResult = cryptoContext->EvalMult(ciphertext1, ciphertext2);
    auto end_mult = std::chrono::high_resolution_clock::now();

    // Sample Program: Step 5: Decryption

    // Decrypt the result of additions
    Plaintext plaintextAddResult;
    auto start_jiemi = std::chrono::high_resolution_clock::now();
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextAddResult, &plaintextAddResult);
    auto end_jiemi = std::chrono::high_resolution_clock::now();
    // Decrypt the result of multiplications
    Plaintext plaintextMultResult;
    cryptoContext->Decrypt(keyPair.secretKey, ciphertextMultResult, &plaintextMultResult);

    auto end_total = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> encrypt_time = end_encrypt - start_encrypt;
    std::chrono::duration<double> add_time = end_add - start_add;
    std::chrono::duration<double> mult_time = end_mult - start_mult;
    std::chrono::duration<double> jiemi_time = end_jiemi - start_jiemi;
    std::chrono::duration<double> total_time = end_total - start_total;

    // std::cout << "Plaintext #1: " << plaintext1 << std::endl;
    // std::cout << "Plaintext #2: " << plaintext2 << std::endl;

    // Output results
    // std::cout << "\nResults of homomorphic computations" << std::endl;
    // std::cout << "#1 + #2 : " << plaintextAddResult << std::endl;
    // std::cout << "#1 * #2 : " << plaintextMultResult << std::endl;

    std::cout << "\nTiming (in milliseconds):" << std::endl;
    std::cout << "Encryption Time: " << encrypt_time.count() * 1000 << " ms" << std::endl;
    std::cout << "Homomorphic Addition Time: " << add_time.count() * 1000 << " ms" << std::endl;
    std::cout << "Homomorphic Multiplication Time: " << mult_time.count() * 1000 << " ms" << std::endl;
    std::cout << "decryption Time: " << jiemi_time.count() * 1000 << " ms" << std::endl;
    std::cout << "Total Program Time: " << total_time.count() * 1000 << " ms" << std::endl;

    return 0;
}