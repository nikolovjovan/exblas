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

vector<double> *elements;
default_random_engine *shuffle_engine;
int *start_indices;
vector<int> *reduction_map;
double *partial_sums;
pthread_barrier_t barrier;
bool result_valid;

typedef struct {
    // Input
    //
    int tid;
    int fpe;
    bool early_exit;
    char type[20];

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

    // ExSum requires array of doubles.
    //
    elements = new vector<double>(element_count);

    int positive_count = 0, negative_count = 0;
    for (int i = 0; i < element_count; ++i) {
        uint32_t bits = (sign_dist(gen) << 31) | (exponent_dist(gen) << 23) | (mantisa_dist(gen) << 0);
        static_assert(sizeof(uint32_t) == sizeof(float));
        float number;
        memcpy(&number, &bits, sizeof(uint32_t));
        (*elements)[i] = number;
        if (!perf_test && print_elements) {
            cout << i + 1 << ". element: " << fixed << setprecision(10) << number << " (" << scientific
                 << setprecision(10) << number << ')' << endl;
            if (number > 0) {
                positive_count++;
            } else {
                negative_count++;
            }
        }
    }

    if (perf_test) {
        return;
    }

    if (print_elements) {
        cout << "Number of positive elements: " << positive_count << endl;
        cout << "Number of negative elements: " << negative_count << endl;
    }

    cout << "Successfully generated " << element_count << " random floating-point numbers." << endl;
}

void run_sequential_reproducible(const char type[], const int fpe = 0, const bool early_exit = false, const bool parallel = false)
{
    chrono::steady_clock::time_point start;
    float result, sum;
    uint64_t time;

    for (int run_idx = 0; run_idx <= repeat_count; ++run_idx) {
        if (run_idx == 0) {
            start = chrono::steady_clock::now();
        }
        sum = static_cast<float> (exsum(
            /* Ng */            element_count,
            /* ag */            elements->data(),
            /* inca */          1,
            /* offset */        0,
            /* fpe */           fpe,
            /* early_exit */    early_exit,
            /* parallel */      parallel));
        if (run_idx == 0) {
            result = sum;
            time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
            if (perf_test) {
                cout << fixed << setprecision(10) << (float) time / 1000.0 << '\t'; // ms
            } else {
                cout << (parallel ? "ExBLAS" : "Sequential") << " reproducible sum (" << type << "): " << fixed << setprecision(10) << result
                    << " (" << scientific << setprecision(10) << result << ')' << endl;
            }
        } else if (sum != result) {
            cout << (parallel ? "ExBLAS" : "Sequential") << " reproducible sum (" << type << ") not reproducible after " << run_idx << " runs!" << endl;
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
        int blk_size = element_count / thread_count;
        int remainder = element_count - thread_count * blk_size;
        start_indices[0] = 0;
        for (int i = 1; i < thread_count; ++i) {
            start_indices[i] = start_indices[i - 1] + (blk_size + (i <= remainder ? 1 : 0));
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
        partial_sums[params->tid] = exsum(
            /* Ng */            end,
            /* ag */            elements->data(),
            /* inca */          1,
            /* offset */        start_indices[params->tid],
            /* fpe */           params->fpe,
            /* early_exit */    params->early_exit,
            /* parallel */      false);

        // Wait on barrier to synchronize all threads to start parallel reduction
        pthread_barrier_wait(&barrier);

        if (params->tid == 0)
        {
            // Reduce partial sums
            // NOTE: This may be inaccurate due to rounding of partial sums to double.
            //       However, it should still be reproducible.
            //
            float result = static_cast<float> (exsum(
                /* Ng */            thread_count,
                /* ag */            partial_sums,
                /* inca */          1,
                /* offset */        0,
                /* fpe */           params->fpe,
                /* early_exit */    params->early_exit,
                /* parallel */      false));

            // Check results
            //
            if (repeat_counter == 0) {
                params->time = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count();
                params->result = result;
                result_valid = true;
                if (perf_test) {
                    cout << fixed << setprecision(10) << (float) params->time / 1000.0 << '\t'; // ms
                } else {
                    cout << "Parallel reproducible sum (" << params->type << "): " << fixed << setprecision(10) << params->result
                         << " (" << scientific << setprecision(10) << params->result << ')' << endl;
                }
            } else if (result != params->result) {
                cout << "Parallel reproducible sum (" << params->type << ") not reproducible after " << repeat_counter << " runs!" << endl;
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

void run_parallel_reproducible(const char type[], const int fpe, const bool early_exit = false)
{
    // Get number of available processors
    const int processor_count = thread::hardware_concurrency();
    // cout << "Processor count: " << processor_count << endl;

    // Initialize POSIX threads
    pthread_t *threads = new pthread_t[thread_count];
    pthread_attr_t attr;
    kernel_params *params = new kernel_params[thread_count];

    start_indices = new int[thread_count];
    partial_sums = new double[thread_count];

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
        strncpy(params[i].type, type, strlen(type));
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

    delete[] partial_sums;
    delete[] start_indices;
    delete[] params;
    delete[] threads;
}

void cleanup()
{
    delete shuffle_engine;
    delete elements;
}

int main(int argc, char *argv[])
{
    parse_parameters(argc, argv);

    // Disable reproducibility testing.
    //
    repeat_count = 0;

    // Force performance testing to disable unnecessary logging and optimize parallel algorithm.
    //
    perf_test = true;

    cout << "unit: [ms]\n\n";

    for (int step = 0; step < 2; ++step)
    {
        if (step == 0) {
            element_count = 100;
            exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
            exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
        } else {
            element_count = 10000000;
            exponent_min = DEFAULT_EXPONENT_MIN_VALUE;
            exponent_max = DEFAULT_EXPONENT_MAX_VALUE;
        }

        for (int i = 0; i < 7; ++i)
        {
            if (step == 0) {
                cout << "n = " << element_count << "\n\n";
            } else {
                cout << "maxabs(e) = " << exponent_max << "\n\n";
            }

            generate_elements();

            cout << "accumulator-only\n\n";

            cout << "seq\t\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("accumulator-only", 0, false);
            cout << '\n';

            cout << "ExBLAS\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("accumulator-only", 0, false, /* ExBLAS parallel */ true);
            cout << '\n';

            for (thread_count = 1; thread_count <= 128; thread_count <<= 1)
            {
                cout << thread_count << "\t\t";
                for (int run = 0; run < 3; ++run) run_parallel_reproducible("accumulator-only", 0, false);
                cout << '\n';
            }

            cout << "\nfpe2\n\n";

            cout << "seq\t\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("fpe2", 2, false);
            cout << '\n';

            cout << "ExBLAS\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("fpe2", 2, false, /* ExBLAS parallel */ true);
            cout << '\n';

            for (thread_count = 1; thread_count <= 128; thread_count <<= 1)
            {
                cout << thread_count << "\t\t";
                for (int run = 0; run < 3; ++run) run_parallel_reproducible("fpe2", 2, false);
                cout << '\n';
            }

            cout << "\nfpe4\n\n";

            cout << "seq\t\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("fpe4", 4, false);
            cout << '\n';

            cout << "ExBLAS\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("fpe4", 4, false, /* ExBLAS parallel */ true);
            cout << '\n';

            for (thread_count = 1; thread_count <= 128; thread_count <<= 1)
            {
                cout << thread_count << "\t\t";
                for (int run = 0; run < 3; ++run) run_parallel_reproducible("fpe4", 4, false);
                cout << '\n';
            }

            cout << "\nfpe8ee\n\n";

            cout << "seq\t\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("fpe8ee", 8, true);
            cout << '\n';

            cout << "ExBLAS\t";

            for (int run = 0; run < 3; ++run) run_sequential_reproducible("fpe8ee", 8, true, /* ExBLAS parallel */ true);
            cout << '\n';

            for (thread_count = 1; thread_count <= 128; thread_count <<= 1)
            {
                cout << thread_count << "\t\t";
                for (int run = 0; run < 3; ++run) run_parallel_reproducible("fpe8ee", 8, true);
                cout << '\n';
            }

            delete elements;

            cout << '\n';
            
            if (step == 0) {
                element_count *= 10;
            } else {
                exponent_min -= 15;
                exponent_max += 15;
            }
        }
    }

    return EXIT_SUCCESS;
}