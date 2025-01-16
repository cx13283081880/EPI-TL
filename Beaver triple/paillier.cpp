#include <iostream>
#include <gmp.h>
#include <chrono>  // 引入计时库
#include <vector>  // 用于存储多对数据
#include <cstdlib> // 用于生成随机数

class Paillier {
public:
    mpz_t n, nsquare, g, lambda, mu;

    Paillier() {
        mpz_inits(n, nsquare, g, lambda, mu, NULL);
    }

    ~Paillier() {
        mpz_clears(n, nsquare, g, lambda, mu, NULL);
    }

    // 密钥生成
    void keygen(int keysize) {
        mpz_t p, q, p_1, q_1, tmp;
        mpz_inits(p, q, p_1, q_1, tmp, NULL);

        gmp_randstate_t state;
        gmp_randinit_default(state);
        gmp_randseed_ui(state, time(NULL));

        mpz_urandomb(tmp, state, keysize / 2);
        mpz_nextprime(p, tmp);

        mpz_urandomb(tmp, state, keysize / 2);
        mpz_nextprime(q, tmp);

        mpz_mul(n, p, q);
        mpz_mul(nsquare, n, n);

        mpz_add_ui(g, n, 1);

        mpz_sub_ui(p_1, p, 1);
        mpz_sub_ui(q_1, q, 1);
        mpz_lcm(lambda, p_1, q_1);

        mpz_t g_lambda;
        mpz_init(g_lambda);
        mpz_powm(g_lambda, g, lambda, nsquare);

        L(mu, g_lambda);
        mpz_mod(mu, mu, n);
        mpz_invert(mu, mu, n);

        mpz_clears(p, q, p_1, q_1, g_lambda, tmp, NULL);
        gmp_randclear(state);
    }

    // 加密
    void encrypt(mpz_t c, mpz_t m, gmp_randstate_t state) {
        mpz_t r, gm, rn;
        mpz_inits(r, gm, rn, NULL);

        mpz_urandomm(r, state, n);

        mpz_powm(gm, g, m, nsquare);
        mpz_powm(rn, r, n, nsquare);
        mpz_mul(c, gm, rn);
        mpz_mod(c, c, nsquare);

        mpz_clears(r, gm, rn, NULL);
    }

    // 解密
    void decrypt(mpz_t m, mpz_t c) {
        mpz_t u;
        mpz_init(u);

        mpz_powm(u, c, lambda, nsquare);
        L(m, u);
        mpz_mul(m, m, mu);
        mpz_mod(m, m, n);

        mpz_clear(u);
    }

    // 同态加法：c1 * c2 mod n^2
    void homomorphic_add(mpz_t result, mpz_t c1, mpz_t c2) {
        mpz_mul(result, c1, c2);      // result = c1 * c2
        mpz_mod(result, result, nsquare);  // result = result mod n^2
    }

    // 同态乘法：c^m mod n^2
    void homomorphic_mult(mpz_t result, mpz_t c, mpz_t m) {
        mpz_powm(result, c, m, nsquare);   // result = c^m mod n^2
    }

private:
    // L函数，计算 L(x) = (x - 1) / n
    void L(mpz_t res, mpz_t x) {
        mpz_sub_ui(res, x, 1);
        mpz_div(res, res, n);
    }
};

int main() {
    Paillier paillier;
    paillier.keygen(3072); // 生成3072位密钥

    // 初始化随机数生成器
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, time(NULL));  // 使用时间作为种子

    mpz_t m1, m2, c1, c2, c_add, c_mult, result1, result2;
    mpz_inits(m1, m2, c1, c2, c_add, c_mult, result1, result2, NULL);

    // 创建50对随机数
    std::vector<std::pair<mpz_t, mpz_t>> pairs(50);
    for (int i = 0; i < 1; ++i) {
        mpz_inits(pairs[i].first, pairs[i].second, NULL);
        mpz_urandomb(pairs[i].first, state, 256); // 生成256位随机数作为m1
        mpz_urandomb(pairs[i].second, state, 256); // 生成256位随机数作为m2
    }

    // 初始化总时间
    double total_encrypt_time = 0.0;
    double total_add_time = 0.0;
    double total_mult_time = 0.0;
    double total_decrypt_time = 0.0;

    // 计时并进行50对测试
    auto start_total = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1; ++i) {
        // 计时加密时间
        auto start_encrypt = std::chrono::high_resolution_clock::now();
        // 加密每一对
        paillier.encrypt(c1, pairs[i].first, state);
        paillier.encrypt(c2, pairs[i].second, state);
        auto end_encrypt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_encrypt = end_encrypt - start_encrypt;
        total_encrypt_time += elapsed_encrypt.count();  // 累加加密时间
        
        // 同态加法：执行3次并计时
        auto start_add = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < 3; ++j) {
            paillier.homomorphic_add(c_add, c1, c2);
        }
        auto end_add = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_add = end_add - start_add;
        total_add_time += elapsed_add.count();  // 累加同态加法时间

        // 同态乘法：执行1次并计时
        auto start_mult = std::chrono::high_resolution_clock::now();
        paillier.homomorphic_mult(c_mult, c1, pairs[i].second);
        auto end_mult = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_mult = end_mult - start_mult;
        total_mult_time += elapsed_mult.count();  // 累加同态乘法时间

        // 解密时间
        auto start_decrypt = std::chrono::high_resolution_clock::now();
        paillier.decrypt(result1, c_add);
        paillier.decrypt(result2, c_mult);
        auto end_decrypt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_decrypt = end_decrypt - start_decrypt;
        total_decrypt_time += elapsed_decrypt.count();  // 累加解密时间

        // 输出每一对的时间
        std::cout << "Pair " << i+1 << " Encryption Time: " << elapsed_encrypt.count() * 1000 << " ms" << std::endl;
        std::cout << "Pair " << i+1 << " Homomorphic Addition Time: " << elapsed_add.count() * 1000 << " ms" << std::endl;
        std::cout << "Pair " << i+1 << " Homomorphic Multiplication Time: " << elapsed_mult.count() * 1000 << " ms" << std::endl;
        std::cout << "Pair " << i+1 << " Decryption Time: " << elapsed_decrypt.count() * 1000 << " ms" << std::endl;
    }

    // 计算并输出总时间
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end_total - start_total;
    std::cout << "\nTotal Time for 50 pairs: " << total_elapsed.count() * 1000 << " ms" << std::endl;
    std::cout << "Total Encryption Time: " << total_encrypt_time * 1000 << " ms" << std::endl;
    std::cout << "Total Homomorphic Addition Time: " << total_add_time * 1000 << " ms" << std::endl;
    std::cout << "Total Homomorphic Multiplication Time: " << total_mult_time * 1000 << " ms" << std::endl;
    std::cout << "Total Decryption Time: " << total_decrypt_time * 1000 << " ms" << std::endl;

    // 清理
    for (int i = 0; i < 1; ++i) {
        mpz_clears(pairs[i].first, pairs[i].second, NULL);
    }

    mpz_clears(m1, m2, c1, c2, c_add, c_mult, result1, result2, NULL);
    gmp_randclear(state);
    
    return 0;
}
