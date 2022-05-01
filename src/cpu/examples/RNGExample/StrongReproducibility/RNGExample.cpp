#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <random>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#include "blas1.hpp"
#include "common.hpp"

using namespace std;

// Floating-point specification constants

constexpr int EXPONENT_MIN_VALUE = -126;
constexpr int EXPONENT_MAX_VALUE = 127;
constexpr int EXPONENT_BIAS = 127;

// Default values

constexpr int DEFAULT_THREAD_COUNT = 8;
constexpr int DEFAULT_ELEMENT_COUNT = 1000;
constexpr uint32_t DEFAULT_SEED = 1549813198;
constexpr int DEFAULT_EXPONENT_MIN_VALUE = -10;
constexpr int DEFAULT_EXPONENT_MAX_VALUE = 10;
constexpr int DEFAULT_REPEAT_COUNT = 100;

// Program parameters

int thread_count = DEFAULT_THREAD_COUNT;
int element_count = DEFAULT_ELEMENT_COUNT;
uint32_t seed = DEFAULT_SEED;
int exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
int exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
int repeat_count = DEFAULT_REPEAT_COUNT;
bool print_elements = false;
bool perf_test = false;

// Shared variables

vector<float> *elements;
vector<double> *dbl_elements;
default_random_engine *shuffle_engine;
int *start_indices;
vector<int> *reduction_map;
float *partial_sums;
double *dbl_partial_sums; // needed for exblas reduction
pthread_barrier_t barrier;
bool result_valid;

// Results

float sum_sequential, sum_parallel;
uint64_t time_sequential, time_parallel;

float sum_sequential_acc, sum_sequential_fpe2, sum_sequential_fpe4, sum_sequential_fpe8ee;
uint64_t time_sequential_acc, time_sequential_fpe2, time_sequential_fpe4, time_sequential_fpe8ee;

float sum_parallel_acc, sum_parallel_fpe2, sum_parallel_fpe4, sum_parallel_fpe8ee;
uint64_t time_parallel_acc, time_parallel_fpe2, time_parallel_fpe4, time_parallel_fpe8ee;

typedef struct {
    // Input
    //
    int tid;
    int fpe;
    bool early_exit;

    // Output
    //
    float result;
    uint64_t time;
} kernel_params;

void print_usage(char program_name[])
{
    cout << "Usage: " << program_name << " <options>" << endl;
    cout << "  -t <num>: Uses <num> threads for parallel execution (default value: 8)" << endl;
    cout << "  -n <num>: Generates <num> random floating-point numbers for analysis (default value: 1000)" << endl;
    cout << "  -s <num>: Generates the numbers using <num> as seed (default value: 1549813198)" << endl;
    cout << "  -l <num>: Uses <num> as exponent minimum value (default value: -10)" << endl;
    cout << "  -h <num>: Uses <num> as exponent maximum value (inclusive) (default value: 10)" << endl;
    cout << "  -r <num>: Repeats the execution of both implementations <num> times for reproducibility study (default value: 100)" << endl;
    cout << "  -p: Print generated numbers before analysis" << endl;
    cout << "  -!: Split elements evenly between threads for performance testing" << endl;
    cout << "  -?: Print this message" << endl;
}

void parse_parameters(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        int arglen = strlen(argv[i]);
        if (arglen < 2 || argv[i][0] != '-') {
            cout << "Invalid argument: \"" << argv[i] << '"' << endl;
            print_usage(argv[0]);
            exit(EXIT_FAILURE);
        }
        int n;
        switch (argv[i][1]) {
        case 't':
            if (i + 1 == argc) {
                cout << "Thread count not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < 1) {
                cout << "Invalid thread count: " << n << "! Using default value: " << DEFAULT_THREAD_COUNT << endl;
            } else {
                thread_count = n;
            }
            break;
        case 'n':
            if (i + 1 == argc) {
                cout << "Element count not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < 1) {
                cout << "Invalid element count: " << n << "! Using default value: " << DEFAULT_ELEMENT_COUNT << endl;
            } else {
                element_count = n;
            }
            break;
        case 's':
            if (i + 1 == argc) {
                cout << "Seed not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n == 0) {
                random_device rd;
                seed = rd();
                cout << "Special seed value: 0. Generated random seed: " << seed << endl;
            } else {
                seed = n;
            }
            break;
        case 'l':
            if (i + 1 == argc) {
                cout << "Exponent minimum value not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < EXPONENT_MIN_VALUE) {
                cout << "Invalid exponent minimum value: " << n
                     << "! Using default value: " << DEFAULT_EXPONENT_MIN_VALUE << endl;
            } else {
                exponent_min = n;
            }
            break;
        case 'h':
            if (i + 1 == argc) {
                cout << "Exponent maximum value not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n > EXPONENT_MAX_VALUE) {
                cout << "Invalid exponent maximum value: " << n
                     << "! Using default value: " << DEFAULT_EXPONENT_MAX_VALUE << endl;
            } else {
                exponent_max = n;
            }
            break;
        case 'r':
            if (i + 1 == argc) {
                cout << "Repeat count not specified!" << endl;
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            n = atoi(argv[++i]);
            if (n < 0) {
                cout << "Invalid repeat count: " << n << "! Using default value: " << DEFAULT_REPEAT_COUNT << endl;
            } else {
                repeat_count = n;
            }
            break;
        case 'p':
            print_elements = true;
            break;
        case '!':
            perf_test = true;
            break;
        case '?':
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        default:
            cout << "Invalid option: \"" << argv[i] << "\"!" << endl;
            exit(EXIT_FAILURE);
        }
    }
    if (thread_count > element_count) {
        cout << "There are more threads available than there are elements to process. Reducing thread count to element count." << endl;
        thread_count = element_count;
    }
}

void print_parameters()
{
    cout << "Parameters:" << endl;
    cout << "  Thread count:             " << thread_count << endl;
    cout << "  Element count:            " << element_count << endl;
    cout << "  Seed:                     " << seed << endl;
    cout << "  Exponent minimum value:   " << exponent_min << endl;
    cout << "  Exponent maximum value:   " << exponent_max << endl;
    cout << "  Repeat count:             " << repeat_count << endl;
    cout << "  Print elements:           " << (print_elements ? "YES" : "NO") << endl;
    cout << "  Performance test:         " << (perf_test ? "YES" : "NO") << endl;
}

void generate_elements()
{
    mt19937 gen(seed);
    uniform_int_distribution<uint32_t> sign_dist(0, 1);
    // uniform_int_distribution<uint32_t> exponent_dist(0, (1 << 8) - 1);
    uniform_int_distribution<uint32_t> exponent_dist(exponent_min + EXPONENT_BIAS, exponent_max + EXPONENT_BIAS);
    uniform_int_distribution<uint32_t> mantisa_dist(0, (1 << 23) - 1);

    elements = new vector<float>(element_count);

    // exsum requires double
    //
    dbl_elements = new vector<double>(element_count);

    int positive_count = 0, negative_count = 0;
    for (int i = 0; i < element_count; ++i) {
        uint32_t bits = (sign_dist(gen) << 31) | (exponent_dist(gen) << 23) | (mantisa_dist(gen) << 0);
        static_assert(sizeof(uint32_t) == sizeof(float));
        float number;
        memcpy(&number, &bits, sizeof(uint32_t));
        (*elements)[i] = number;
        (*dbl_elements)[i] = static_cast<double> (number);
        if (print_elements) {
            cout << i + 1 << ". element: " << fixed << setprecision(10) << number << " (" << scientific
                 << setprecision(10) << number << ')' << endl;
            if (number > 0) {
                positive_count++;
            } else {
                negative_count++;
            }
        }
    }

    if (print_elements) {
        cout << "Number of positive elements: " << positive_count << endl;
        cout << "Number of negative elements: " << negative_count << endl;
    }

    cout << "Successfully generated " << element_count << " random floating-point numbers." << endl;
}

void run_sequential(float& result, uint64_t& time)
{
    chrono::steady_clock::time_point start;
    float sum;
    for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
        if (run_idx == 0) {
            start = chrono::steady_clock::now();
        }
        sum = 0;
        for (int i = 0; i < element_count; ++i) {
            // NOTE: This operation results in non-reproducible results. The reason is that
            //       element order is shuffled after every repetition hence rounding errors
            //       and other inherent errors with floating-point arithmetic occur.
            sum += (*elements)[i];
        }
        if (run_idx == 0) {
            result = sum;
            time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
            cout << "Sequential sum: " << fixed << setprecision(10) << result << " (" << scientific
                 << setprecision(10) << result << ')' << endl;
        } else if (sum != result) {
            cout << "Sequential sum not reproducible after " << run_idx << " runs!" << endl;
            break;
        }
        if (run_idx < repeat_count) {
            // Shuffle element vector to incur variability
            shuffle(elements->begin(), elements->end(), *shuffle_engine);
        }
    }
}

void run_sequential_reproducible(float& result, uint64_t& time, const int fpe = 0, const bool early_exit = false)
{
    chrono::steady_clock::time_point start;
    float sum;
    for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
        if (run_idx == 0) {
            start = chrono::steady_clock::now();
        }
        sum = static_cast<float> (exsum(element_count, dbl_elements->data(), 1, 0, fpe, early_exit));
        if (run_idx == 0) {
            result = sum;
            time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
            cout << "Sequential sum (reproducible): " << fixed << setprecision(10) << result
                 << " (" << scientific << setprecision(10) << result << ')' << endl;
        } else if (sum != result) {
            cout << "Sequential sum not reproducible after " << run_idx << " runs!" << endl;
            break;
        }
        if (run_idx < repeat_count) {
            // Shuffle element vector to incur variability
            shuffle(elements->begin(), elements->end(), *shuffle_engine);
            // auto time_loop = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() -
            // start).count(); cout << "Run " << run_idx << ". passed! Time elapsed: " << time_loop << " [us] (" <<
            // fixed << setprecision(10) << (float) time_loop / 1000.0 << " [ms])" << endl;
        }
    }
}

void generate_start_indices()
{
    if (perf_test) {
        int blk_size = (element_count + thread_count - 1) / thread_count;
        for (int i = 0; i < thread_count; ++i) {
            start_indices[i] = i * blk_size;
        }
        return;
    }
    int min = 1;
    int max = element_count - thread_count * min;
    if (max == 0) {
        for (int i = 0; i < thread_count; ++i) {
            start_indices[i] = i;
        }
        return;
    }
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> size_dist(0, max);
    start_indices[0] = 0;
    for (int i = 1; i < thread_count; ++i) {
        start_indices[i] = size_dist(gen);
    }
    sort(start_indices, start_indices + thread_count);
    for (int i = 1; i < thread_count; ++i) {
        start_indices[i - 1] = start_indices[i] - start_indices[i - 1] + min;
    }
    start_indices[thread_count - 1] = max - start_indices[thread_count - 1] + min;
    int size = 0;
    for (int i = 0; i < thread_count; ++i) {
        int tmp = start_indices[i];
        start_indices[i] = size;
        size += tmp;
    }
}

void *kernel_sum(void *data)
{
    kernel_params* params = static_cast<kernel_params*> (data);
    chrono::steady_clock::time_point start;
    for (int repeat_counter = 0; repeat_counter <= repeat_count; ++repeat_counter) {
        if (params->tid == 0) {
            // Generate start indices
            generate_start_indices();
            if (repeat_counter == 0) {
                start = chrono::steady_clock::now();
            }
        }

        // Synchronize on barrier to get start index
        pthread_barrier_wait(&barrier);

        partial_sums[params->tid] = 0;
        int end = params->tid < thread_count - 1 ? start_indices[params->tid + 1] : element_count;
        for (int i = start_indices[params->tid]; i < end; ++i) {
            partial_sums[params->tid] += (*elements)[i];
        }

        // Wait on barrier to synchronize all threads to start parallel reduction
        pthread_barrier_wait(&barrier);

        // Reduce partial sums
        int pow2_count = 1, step_count = 0;
        while (pow2_count < thread_count) {
            pow2_count <<= 1;
            step_count++;
        }
        pow2_count >>= 1;
        if (step_count > 0) {
            // First reduction step is not based on power of two reduction since thread_count may not be a power of
            // two...
            if (params->tid < thread_count - pow2_count) {
                partial_sums[(*reduction_map)[params->tid]] += partial_sums[(*reduction_map)[params->tid + pow2_count]];
            }
            // Wait on barrier to synchronize all threads for next reduction step
            pthread_barrier_wait(&barrier);
            // The rest of the steps are simple power of two reduction. There are now pow2_count partial sums to
            // reduce...
            for (int i = 1; i < step_count; ++i) {
                pow2_count >>= 1;
                if (params->tid < pow2_count) {
                    partial_sums[(*reduction_map)[params->tid]] += partial_sums[(*reduction_map)[params->tid + pow2_count]];
                }
                // Wait on barrier to synchronize all threads for next reduction step
                pthread_barrier_wait(&barrier);
            }
        }

        // Check results
        if (params->tid == 0) {
            if (repeat_counter == 0) {
                params->time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
                params->result = partial_sums[(*reduction_map)[params->tid]];
                result_valid = true;
                cout << "Parallel sum: " << fixed << setprecision(10) << params->result << " (" << scientific
                     << setprecision(10) << params->result << ')' << endl;
            } else if (partial_sums[(*reduction_map)[params->tid]] != params->result) {
                cout << "Parallel sum not reproducible after " << repeat_counter << " runs!" << endl;
                result_valid = false;
            }
            if (repeat_counter < repeat_count && result_valid) {
                // Shuffle element vector to incur variability
                shuffle(elements->begin(), elements->end(), *shuffle_engine);
                // Shuffle reduction map to incur additional variability
                shuffle(reduction_map->begin(), reduction_map->end(), *shuffle_engine);
            }
        }

        // Wait on barrier to synchronize all threads for next repetition
        pthread_barrier_wait(&barrier);
        if (!result_valid)
            break;
    }
    pthread_exit(NULL);
}

void run_parallel(float& result, uint64_t& time)
{
    // Get number of available processors
    const int processor_count = thread::hardware_concurrency();
    // cout << "Processor count: " << processor_count << endl;

    // Initialize POSIX threads
    pthread_t *threads = new pthread_t[thread_count];
    pthread_attr_t attr;
    kernel_params *params = new kernel_params[thread_count];

    start_indices = new int[thread_count];
    reduction_map = new vector<int>(thread_count);
    partial_sums = new float[thread_count];

    int err;

    // Create thread attribute object (in case this code is used with compilers other than G++)
    err = pthread_attr_init(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Create thread barrier
    err = pthread_barrier_init(&barrier, NULL, thread_count);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Create threads with affinity
    cpu_set_t cpus;
    for (int i = 0; i < thread_count; ++i) {
        params[i].tid = i;       // Generate thread ids for easy work sharing
        (*reduction_map)[i] = i; // Generate initial reduction map for each thread id (same as thread id)
        CPU_ZERO(&cpus);
        CPU_SET(i % processor_count, &cpus);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
        err = pthread_create(&(threads[i]), &attr, kernel_sum, &(params[i]));
        if (err) {
            fprintf(stderr, "Error - pthread_create() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    // Wait for other threads to complete
    for (int i = 0; i < thread_count; ++i) {
        err = pthread_join(threads[i], NULL);
        if (err) {
            fprintf(stderr, "Error - pthread_join() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    // Cleanup
    err = pthread_barrier_destroy(&barrier);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = pthread_attr_destroy(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    result = params->result;
    time = params->time;

    delete[] partial_sums;
    delete reduction_map;
    delete[] start_indices;
    delete[] params;
    delete[] threads;
}

void *kernel_sum_reproducible(void *data)
{
    kernel_params *params = static_cast<kernel_params*> (data);
    chrono::steady_clock::time_point start;
    for (int repeat_counter = 0; repeat_counter <= repeat_count; ++repeat_counter) {
        if (params->tid == 0) {
            // Generate start indices
            generate_start_indices();
            if (repeat_counter == 0) {
                start = chrono::steady_clock::now();
            }
        }

        // Synchronize on barrier to get start index
        pthread_barrier_wait(&barrier);

        int end = params->tid < thread_count - 1 ? start_indices[params->tid + 1] : element_count;
        dbl_partial_sums[params->tid] = exsum(end - start_indices[params->tid], dbl_elements->data(), 1, start_indices[params->tid], params->fpe, params->early_exit);

        // Wait on barrier to synchronize all threads to start parallel reduction
        pthread_barrier_wait(&barrier);

        if (params->tid == 0)
        {
            // Reduce partial sums
            dbl_partial_sums[0] = exsum(thread_count, dbl_partial_sums, 1, 0, params->fpe, params->early_exit);
            // Check results
            if (repeat_counter == 0) {
                params->time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
                params->result = static_cast<float> (partial_sums[0]);
                result_valid = true;
                cout << "Parallel sum (reproducible): " << fixed << setprecision(10) << params->result
                     << " (" << scientific << setprecision(10) << params->result << ')' << endl;
            } else if (static_cast<float> (partial_sums[0]) != params->result) {
                cout << "Parallel sum (reproducible) not reproducible after " << repeat_counter << " runs!" << endl;
                result_valid = false;
            }
            if (repeat_counter < repeat_count && result_valid) {
                // Shuffle element vector to incur variability
                shuffle(elements->begin(), elements->end(), *shuffle_engine);
            }
        }

        // Wait on barrier to synchronize all threads for next repetition
        pthread_barrier_wait(&barrier);
        if (!result_valid)
            break;
    }
    pthread_exit(NULL);
}

void run_parallel_reproducible(float& result, uint64_t& time, const int fpe, const bool early_exit = false)
{
    // Get number of available processors
    const int processor_count = thread::hardware_concurrency();
    // cout << "Processor count: " << processor_count << endl;

    // Initialize POSIX threads
    pthread_t *threads = new pthread_t[thread_count];
    pthread_attr_t attr;
    kernel_params *params = new kernel_params[thread_count];

    start_indices = new int[thread_count];
    dbl_partial_sums = new double[thread_count];

    int err;

    // Create thread attribute object (in case this code is used with compilers other than G++)
    err = pthread_attr_init(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Create thread barrier
    err = pthread_barrier_init(&barrier, NULL, thread_count);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_init() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Create threads with affinity
    cpu_set_t cpus;
    for (int i = 0; i < thread_count; ++i) {
        params[i].tid = i;   // Generate thread ids for easy work sharing
        params[i].fpe = fpe;
        params[i].early_exit = early_exit;
        CPU_ZERO(&cpus);
        CPU_SET(i % processor_count, &cpus);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
        err = pthread_create(&(threads[i]), &attr, kernel_sum_reproducible, &(params[i]));
        if (err) {
            fprintf(stderr, "Error - pthread_create() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    // Wait for other threads to complete
    for (int i = 0; i < thread_count; ++i) {
        err = pthread_join(threads[i], NULL);
        if (err) {
            fprintf(stderr, "Error - pthread_join() return code: %d\n", err);
            exit(EXIT_FAILURE);
        }
    }

    // Cleanup
    err = pthread_barrier_destroy(&barrier);
    if (err) {
        fprintf(stderr, "Error - pthread_barrier_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = pthread_attr_destroy(&attr);
    if (err) {
        fprintf(stderr, "Error - pthread_attr_destroy() return code: %d\n", err);
        exit(EXIT_FAILURE);
    }

    result = params->result;
    time = params->time;

    delete[] dbl_partial_sums;
    delete[] start_indices;
    delete[] params;
    delete[] threads;
}

void print_diff_nrr(const char parallel[], const char type[], const float sum_normal, const float sum_reproducible)
{    
    if (sum_normal != sum_reproducible) {
        cout << "Non-reproducible and reproducible " << parallel << " (" << type << ") sums do not match!" << endl;
    }
}

void print_diff_reproducible(const char type[], const float sum1, const float sum2)
{
    cout << "Reproducible sequential and parallel (" << type << ") sums "
         << (sum1 != sum2 ? "do not " : "") << "match!" << endl;
}

void print_time_speedup(const char type[], const uint64_t time_seq, const uint64_t time_par)
{
    stringstream buffer;
    if (strlen(type) > 0)
    {
        buffer << " (reproducible - " << type << ")";
    }
    cout << endl
         << "Sequential execution time" << buffer.str() << ": " << time_seq << " [us] (" << fixed << setprecision(10)
         << (float) time_seq / 1000.0 << " [ms])" << endl;
    cout << "Parallel execution time" << buffer.str() << ": " << time_par << " [us] (" << fixed << setprecision(10)
         << (float) time_par / 1000.0 << " [ms])" << endl;
    cout << "Speedup" << buffer.str() << ": " << fixed << setprecision(10) << ((float) time_seq) / ((float) time_par) << endl
         << endl;
}

void print_time_diff(const char type[], const uint64_t time_seq, const uint64_t time_par, const uint64_t time_seq_rep, const uint64_t time_par_rep)
{
    cout << "Time sequential reproducible - " << type << " / non-reproducible: " << fixed << setprecision(10)
         << ((float) time_seq_rep) / ((float) time_seq) << endl;
    cout << "Time parallel reproducible - " << type << " / non-reproducible: " << fixed << setprecision(10)
         << ((float) time_par_rep) / ((float) time_par) << endl;
}

void cleanup()
{
    delete shuffle_engine;
    delete elements;
}

int main(int argc, char *argv[])
{
    parse_parameters(argc, argv);
    print_parameters();

    generate_elements();

    shuffle_engine = new default_random_engine(seed);

    run_sequential(sum_sequential, time_sequential);
    run_parallel(sum_parallel, time_parallel);

    run_sequential_reproducible(sum_sequential_acc,     time_sequential_acc,    0, false);
    run_sequential_reproducible(sum_sequential_fpe2,    time_sequential_fpe2,   2, false);
    run_sequential_reproducible(sum_sequential_fpe4,    time_sequential_fpe4,   4, false);
    run_sequential_reproducible(sum_sequential_fpe8ee,  time_sequential_fpe8ee, 8, true);

    run_parallel_reproducible(sum_parallel_acc,     time_parallel_acc,    0, false);
    run_parallel_reproducible(sum_parallel_fpe2,    time_parallel_fpe2,   2, false);
    run_parallel_reproducible(sum_parallel_fpe4,    time_parallel_fpe4,   4, false);
    run_parallel_reproducible(sum_parallel_fpe8ee,  time_parallel_fpe8ee, 8, true);

    print_diff_nrr("sequential",    "accumulator-only", sum_sequential, sum_sequential_acc);
    print_diff_nrr("sequential",    "fpe2",             sum_sequential, sum_sequential_fpe2);
    print_diff_nrr("sequential",    "fpe4",             sum_sequential, sum_sequential_fpe4);
    print_diff_nrr("sequential",    "fpe8ee",           sum_sequential, sum_sequential_fpe8ee);
    print_diff_nrr("parallel",      "accumulator-only", sum_parallel,   sum_sequential_acc);
    print_diff_nrr("parallel",      "fpe2",             sum_parallel,   sum_sequential_fpe2);
    print_diff_nrr("parallel",      "fpe4",             sum_parallel,   sum_sequential_fpe4);
    print_diff_nrr("parallel",      "fpe8ee",           sum_parallel,   sum_sequential_fpe8ee);

    cout << endl;

    print_diff_reproducible("accumulator-only", sum_sequential_acc,     sum_parallel_acc);
    print_diff_reproducible("fpe2",             sum_sequential_fpe2,    sum_parallel_fpe2);
    print_diff_reproducible("fpe4",             sum_sequential_fpe4,    sum_parallel_fpe4);
    print_diff_reproducible("fpe8ee",           sum_sequential_fpe8ee,  sum_parallel_fpe8ee);

    cout << endl;

    print_time_speedup("", time_sequential, time_parallel);

    print_time_speedup  ("acc", time_sequential_acc, time_parallel_acc);
    print_time_diff     ("acc", time_sequential, time_parallel, time_sequential_acc, time_parallel_acc);
    
    print_time_speedup  ("fpe2", time_sequential_fpe2, time_parallel_fpe2);
    print_time_diff     ("fpe2", time_sequential, time_parallel, time_sequential_fpe2, time_parallel_fpe2);
    
    print_time_speedup  ("fpe4", time_sequential_fpe4, time_parallel_fpe4);
    print_time_diff     ("fpe4", time_sequential, time_parallel, time_sequential_fpe4, time_parallel_fpe4);
    
    print_time_speedup  ("fpe8ee", time_sequential_fpe8ee, time_parallel_fpe8ee);
    print_time_diff     ("fpe8ee", time_sequential, time_parallel, time_sequential_fpe8ee, time_parallel_fpe8ee);

    cleanup();

    return EXIT_SUCCESS;
}