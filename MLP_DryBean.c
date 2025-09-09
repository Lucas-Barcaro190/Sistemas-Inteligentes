#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

#define MAX_EPOCHS 200
#define LEARNING_RATE 0.001
#define TOTAL_SAMPLES 13611
#define NUM_INPUTS 16
#define NUM_CLASSES 7

typedef enum { SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON, UNKNOWN } DryBeanSpecies;

typedef struct {
    double *weights;
    double bias;
    double output;   // pós-ativação
    double z;        // pré-ativação
    double delta;
    int numInputs;
} Neuron;

typedef struct {
    Neuron *neurons;
    int numNeurons;
} Layer;

typedef struct {
    Layer *layers;
    int numLayers;
} MLP;

// --- Prototypes ---
MLP* createMLP(int numLayers, const int neuronsPerLayer[]);
void freeMLP(MLP *net);
void forwardPass(MLP *net, double inputs[]);
void backwardPass(MLP *net, double targets[]);
void updateWeights(MLP *net, double inputs[]);
void shuffleData(int n, double **data, double **targets);
void normalizeData(double **data);
double calculateAccuracy(MLP *net, int numSamples, double **data, double **targets);
double calculateLoss(MLP *net, int numSamples, double **data, double **targets);
double calculateTrainLoss(MLP *net, int numSamples, double **data, double **targets);

double leakyRelu(double x) { return x > 0 ? x : 0.01 * x; }
double leakyReluDerivative(double x) { return x > 0 ? 1.0 : 0.01; }
void softmax(Layer *outputLayer);
DryBeanSpecies speciesNameToEnum(char *species);

// --- MAIN ---
int main() {
    srand(time(NULL));

    double **allData = malloc(TOTAL_SAMPLES * sizeof(double*));
    double **allTargets = malloc(TOTAL_SAMPLES * sizeof(double*));
    for(int i = 0; i < TOTAL_SAMPLES; ++i) {
        allData[i] = malloc(NUM_INPUTS * sizeof(double));
        allTargets[i] = calloc(NUM_CLASSES, sizeof(double));
    }

    FILE *file = fopen("dry_bean_dataset.csv", "r");
    if (!file) {
        printf("Error: dry_bean_dataset.csv not found.\n");
        return 1;
    }

    char line[512];
    int count = 0;
    fgets(line, sizeof(line), file); // skip header

    while (fgets(line, sizeof(line), file) && count < TOTAL_SAMPLES) {
        char *token = strtok(line, ",");
        for(int i = 0; i < NUM_INPUTS; i++) {
            allData[count][i] = atof(token);
            token = strtok(NULL, ",");
        }
        if (token) {
            token[strcspn(token, "\r\n")] = 0;
            DryBeanSpecies species = speciesNameToEnum(token);
            if (species != UNKNOWN) {
                allTargets[count][species] = 1.0;
            }
        }
        count++;
    }
    fclose(file);

    normalizeData(allData);
    printf("Dataset normalized successfully.\n");
    shuffleData(TOTAL_SAMPLES, allData, allTargets);
    printf("Dataset shuffled randomly.\n");

    int numTrainingSamples = (int)(TOTAL_SAMPLES * 0.8);
    int numValidationSamples = TOTAL_SAMPLES - numTrainingSamples;
    double **trainingData = allData;
    double **trainingTargets = allTargets;
    double **validationData = &allData[numTrainingSamples];
    double **validationTargets = &allTargets[numTrainingSamples];

    printf("Data split into %d training samples and %d validation samples.\n", numTrainingSamples, numValidationSamples);

    const int architecture[] = {NUM_INPUTS, 20, 20, NUM_CLASSES};
    int numArchLayers = 4;

    MLP *net = createMLP(numArchLayers, architecture);
    printf("Training MLP with Leaky ReLU and Softmax Output...\n");

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        for (int i = 0; i < numTrainingSamples; i++) {
            forwardPass(net, trainingData[i]);
            backwardPass(net, trainingTargets[i]);
            updateWeights(net, trainingData[i]);

            // Debug: imprime softmax de 3 amostras na 1ª época
            if (epoch == 0 && i < 3) {
                printf("Sample %d softmax outputs: ", i);
                for (int j = 0; j < NUM_CLASSES; j++) {
                    printf("%.4f ", net->layers[net->numLayers - 1].neurons[j].output);
                }
                printf("\n");
            }
        }

        if ((epoch + 1) % 10 == 0) {
            double trainLoss = calculateTrainLoss(net, numTrainingSamples, trainingData, trainingTargets);
            double valLoss = calculateLoss(net, numValidationSamples, validationData, validationTargets);
            double accuracy = calculateAccuracy(net, numValidationSamples, validationData, validationTargets);

            printf("Epoch %3d/%d - Val Acc: %5.2f%% | Train Loss: %.4f | Val Loss: %.4f\n",
                   epoch + 1, MAX_EPOCHS, accuracy, trainLoss, valLoss);
        }
    }

    freeMLP(net);
    for(int i = 0; i < TOTAL_SAMPLES; ++i) {
        free(allData[i]);
        free(allTargets[i]);
    }
    free(allData);
    free(allTargets);
    printf("Training complete.\n");
    return 0;
}

// --- MLP Functions ---
MLP* createMLP(int numLayers, const int neuronsPerLayer[]) {
    MLP *net = (MLP*)malloc(sizeof(MLP));
    net->numLayers = numLayers - 1;
    net->layers = (Layer*)malloc(net->numLayers * sizeof(Layer));

    for (int i = 0; i < net->numLayers; i++) {
        net->layers[i].numNeurons = neuronsPerLayer[i + 1];
        net->layers[i].neurons = (Neuron*)malloc(net->layers[i].numNeurons * sizeof(Neuron));
        int numInputs = neuronsPerLayer[i];
        for (int j = 0; j < net->layers[i].numNeurons; j++) {
            Neuron *n = &net->layers[i].neurons[j];
            n->numInputs = numInputs;
            n->weights = (double*)malloc(numInputs * sizeof(double));
            double stdDev = sqrt(2.0 / numInputs);
            for (int k = 0; k < numInputs; k++) {
                double u1 = (double)rand() / RAND_MAX;
                double u2 = (double)rand() / RAND_MAX;
                double rand_std_normal = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
                n->weights[k] = rand_std_normal * stdDev;
            }
            n->bias = ((double)rand() / RAND_MAX - 0.5) * 0.01;
        }
    }
    return net;
}

void freeMLP(MLP *net) {
    for (int i = 0; i < net->numLayers; i++) {
        for (int j = 0; j < net->layers[i].numNeurons; j++) {
            free(net->layers[i].neurons[j].weights);
        }
        free(net->layers[i].neurons);
    }
    free(net->layers);
    free(net);
}

void forwardPass(MLP *net, double inputs[]) {
    for (int i = 0; i < net->numLayers; i++) {
        Layer *prev = (i == 0) ? NULL : &net->layers[i - 1];
        Layer *curr = &net->layers[i];
        for (int j = 0; j < curr->numNeurons; j++) {
            double sum = curr->neurons[j].bias;
            for (int k = 0; k < curr->neurons[j].numInputs; k++) {
                double val = (i == 0) ? inputs[k] : prev->neurons[k].output;
                sum += curr->neurons[j].weights[k] * val;
            }
            curr->neurons[j].z = sum;
            curr->neurons[j].output = sum;
        }
        if (i < net->numLayers - 1) {
            for (int j = 0; j < curr->numNeurons; j++) {
                curr->neurons[j].output = leakyRelu(curr->neurons[j].z);
            }
        } else {
            softmax(curr);
        }
    }
}

void softmax(Layer *outputLayer) {
    double maxVal = -DBL_MAX;
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        if (outputLayer->neurons[i].output > maxVal) {
            maxVal = outputLayer->neurons[i].output;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        outputLayer->neurons[i].output = exp(outputLayer->neurons[i].output - maxVal);
        sum += outputLayer->neurons[i].output;
    }

    for (int i = 0; i < outputLayer->numNeurons; i++) {
        outputLayer->neurons[i].output /= sum;
    }
}

void backwardPass(MLP *net, double targets[]) {
    Layer *outputLayer = &net->layers[net->numLayers - 1];

    for (int i = 0; i < outputLayer->numNeurons; i++) {
        double output = outputLayer->neurons[i].output;
        double target = targets[i];
        outputLayer->neurons[i].delta = output - target;
    }

    for (int i = net->numLayers - 2; i >= 0; i--) {
        Layer *curr = &net->layers[i];
        Layer *next = &net->layers[i + 1];
        for (int j = 0; j < curr->numNeurons; j++) {
            double sum = 0.0;
            for (int k = 0; k < next->numNeurons; k++) {
                sum += next->neurons[k].weights[j] * next->neurons[k].delta;
            }
            curr->neurons[j].delta = sum * leakyReluDerivative(curr->neurons[j].z);
        }
    }
}

void updateWeights(MLP *net, double inputs[]) {
    for (int i = 0; i < net->numLayers; i++) {
        Layer *curr = &net->layers[i];
        Layer *prev = (i == 0) ? NULL : &net->layers[i - 1];
        for (int j = 0; j < curr->numNeurons; j++) {
            for (int k = 0; k < curr->neurons[j].numInputs; k++) {
                double input = (i == 0) ? inputs[k] : prev->neurons[k].output;
                curr->neurons[j].weights[k] -= LEARNING_RATE * curr->neurons[j].delta * input;
            }
            curr->neurons[j].bias -= LEARNING_RATE * curr->neurons[j].delta;
        }
    }
}

double calculateAccuracy(MLP *net, int numSamples, double **data, double **targets) {
    int correct = 0;
    for (int i = 0; i < numSamples; i++) {
        forwardPass(net, data[i]);

        int predicted = 0;
        double maxOut = net->layers[net->numLayers - 1].neurons[0].output;
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (net->layers[net->numLayers - 1].neurons[j].output > maxOut) {
                maxOut = net->layers[net->numLayers - 1].neurons[j].output;
                predicted = j;
            }
        }

        int actual = -1;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (targets[i][j] == 1.0) {
                actual = j;
                break;
            }
        }

        if (predicted == actual) correct++;
    }
    return (double)correct / numSamples * 100.0;
}

double calculateLoss(MLP *net, int numSamples, double **data, double **targets) {
    double totalLoss = 0.0;
    for (int i = 0; i < numSamples; i++) {
        forwardPass(net, data[i]);
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (targets[i][j] == 1.0) {
                totalLoss -= log(net->layers[net->numLayers - 1].neurons[j].output + 1e-9);
            }
        }
    }
    return totalLoss / numSamples;
}

double calculateTrainLoss(MLP *net, int numSamples, double **data, double **targets) {
    double totalLoss = 0.0;
    for (int i = 0; i < numSamples; i++) {
        forwardPass(net, data[i]);
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (targets[i][j] == 1.0) {
                totalLoss -= log(net->layers[net->numLayers - 1].neurons[j].output + 1e-9);
            }
        }
    }
    return totalLoss / numSamples;
}

void shuffleData(int n, double **data, double **targets) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        double *tmpData = data[i];
        data[i] = data[j];
        data[j] = tmpData;

        double *tmpTarget = targets[i];
        targets[i] = targets[j];
        targets[j] = tmpTarget;
    }
}

void normalizeData(double **data) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        double mean = 0.0, stddev = 0.0;
        for (int j = 0; j < TOTAL_SAMPLES; j++) {
            mean += data[j][i];
        }
        mean /= TOTAL_SAMPLES;

        for (int j = 0; j < TOTAL_SAMPLES; j++) {
            stddev += (data[j][i] - mean) * (data[j][i] - mean);
        }
        stddev = sqrt(stddev / TOTAL_SAMPLES);

        for (int j = 0; j < TOTAL_SAMPLES; j++) {
            if (stddev > 0.0)
                data[j][i] = (data[j][i] - mean) / stddev;
            else
                data[j][i] = 0.0;
        }
    }
}

DryBeanSpecies speciesNameToEnum(char *species) {
    // transforma em maiúsculo
    for (char *p = species; *p; p++) {
        if (*p >= 'a' && *p <= 'z') {
            *p = *p - 'a' + 'A';
        }
    }

    if (strcmp(species, "SEKER") == 0) return SEKER;
    if (strcmp(species, "BARBUNYA") == 0) return BARBUNYA;
    if (strcmp(species, "BOMBAY") == 0) return BOMBAY;
    if (strcmp(species, "CALI") == 0) return CALI;
    if (strcmp(species, "HOROZ") == 0) return HOROZ;
    if (strcmp(species, "SIRA") == 0) return SIRA;
    if (strcmp(species, "DERMASON") == 0) return DERMASON;
    return UNKNOWN;
}	
