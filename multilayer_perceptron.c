#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_EPOCHS 300
#define LEARNING_RATE 0.1
#define TOTAL_SAMPLES 150

// Enum for species
typedef enum { SETOSA, VERSICOLOR, VIRGINICA } IrisSpecies;

// --- Data Structures for the MLP ---
typedef struct {
    double *weights;
    double bias;
    double output;
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

// --- Function Prototypes ---
MLP* createMLP(int numLayers, const int neuronsPerLayer[]);
void freeMLP(MLP *net);
void forwardPass(MLP *net, double inputs[]);
void backwardPass(MLP *net, double targets[]);
void updateWeights(MLP *net, double inputs[]);
double* runTrainingSession(const char* sessionTitle, int numLayers, const int neuronsPerLayer[],
                           double allData[][4], IrisSpecies allLabels[],
                           IrisSpecies class1, IrisSpecies class2, IrisSpecies class3, int isMultiClass);
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoidDerivative(double x) { return x * (1.0 - x); }
IrisSpecies speciesNameToEnum(char *species);

// --- Main Program Execution ---
int main() {
    srand(time(NULL));

    double allData[TOTAL_SAMPLES][4];
    IrisSpecies allLabels[TOTAL_SAMPLES];
    FILE *file = fopen("iris.data", "r");
    if (!file) { printf("Error: iris.data not found.\n"); return 1; }
    char line[100]; int count = 0;
    while (fgets(line, sizeof(line), file)) {
        char name[20];
        sscanf(line, "%lf,%lf,%lf,%lf,%s", &allData[count][0], &allData[count][1], &allData[count][2], &allData[count][3], name);
        allLabels[count++] = speciesNameToEnum(name);
    }
    fclose(file);

    const int binaryArchitecture[] = {4, 5, 1}; 
    const int multiClassArchitecture[] = {4, 7, 3}; 

    printf("Running all 7 MLP training sessions...\n");
    double *r1 = runTrainingSession("Setosa vs Rest", 3, binaryArchitecture, allData, allLabels, SETOSA, VERSICOLOR, VIRGINICA, 0);
    double *r2 = runTrainingSession("Versicolor vs Rest", 3, binaryArchitecture, allData, allLabels, VERSICOLOR, SETOSA, VIRGINICA, 0);
    double *r3 = runTrainingSession("Virginica vs Rest", 3, binaryArchitecture, allData, allLabels, VIRGINICA, SETOSA, VERSICOLOR, 0);
    double *r4 = runTrainingSession("Setosa vs Versicolor", 3, binaryArchitecture, allData, allLabels, SETOSA, VERSICOLOR, -1, 0);
    double *r5 = runTrainingSession("Setosa vs Virginica", 3, binaryArchitecture, allData, allLabels, SETOSA, VIRGINICA, -1, 0);
    double *r6 = runTrainingSession("Versicolor vs Virginica", 3, binaryArchitecture, allData, allLabels, VERSICOLOR, VIRGINICA, -1, 0);
    double *r7 = runTrainingSession("Multi-Class (All 3)", 3, multiClassArchitecture, allData, allLabels, SETOSA, VERSICOLOR, VIRGINICA, 1);

    FILE *resultsFile = fopen("multilayer_perceptron_results.csv", "w");
    if (!resultsFile) { printf("Error creating results file.\n"); return 1; }
    
    fprintf(resultsFile, "Epoch,TrainingScenario,Accuracy\n");
    const char *scenarios[] = {"Setosa vs Rest", "Versicolor vs Rest", "Virginica vs Rest", "Setosa vs Versicolor", "Setosa vs Virginica", "Versicolor vs Virginica", "Multi-Class (All 3)"};
    double *results[] = {r1, r2, r3, r4, r5, r6, r7};

    for (int i = 0; i < MAX_EPOCHS; i++) {
        for (int j = 0; j < 7; j++) {
            fprintf(resultsFile, "%d,%s,%.2f\n", i + 1, scenarios[j], results[j][i]);
        }
    }
    
    fclose(resultsFile);
    printf("Training complete. Results exported to multilayer_perceptron_results.csv\n");

    for(int i=0; i<7; ++i) free(results[i]);
    return 0;
}

// --- Core MLP Implementation ---

MLP* createMLP(int numLayers, const int neuronsPerLayer[]) {
    MLP *net = (MLP*)malloc(sizeof(MLP));
    net->numLayers = numLayers - 1; // Number of layers with weights (input layer doesn't count)
    net->layers = (Layer*)malloc(net->numLayers * sizeof(Layer));

    for (int i = 0; i < net->numLayers; i++) {
        net->layers[i].numNeurons = neuronsPerLayer[i + 1];
        net->layers[i].neurons = (Neuron*)malloc(net->layers[i].numNeurons * sizeof(Neuron));
        int numInputs = neuronsPerLayer[i];
        for (int j = 0; j < net->layers[i].numNeurons; j++) {
            net->layers[i].neurons[j].numInputs = numInputs;
            net->layers[i].neurons[j].weights = (double*)malloc(numInputs * sizeof(double));
            for (int k = 0; k < numInputs; k++) {
                net->layers[i].neurons[j].weights[k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
            net->layers[i].neurons[j].bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
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
    // First hidden layer
    Layer *firstLayer = &net->layers[0];
    for (int j = 0; j < firstLayer->numNeurons; j++) {
        double activation = firstLayer->neurons[j].bias;
        for (int k = 0; k < firstLayer->neurons[j].numInputs; k++) {
            activation += firstLayer->neurons[j].weights[k] * inputs[k];
        }
        firstLayer->neurons[j].output = sigmoid(activation);
    }

    // Subsequent layers
    for (int i = 1; i < net->numLayers; i++) {
        Layer *prevLayer = &net->layers[i - 1];
        Layer *currentLayer = &net->layers[i];
        for (int j = 0; j < currentLayer->numNeurons; j++) {
            double activation = currentLayer->neurons[j].bias;
            for (int k = 0; k < prevLayer->numNeurons; k++) {
                activation += currentLayer->neurons[j].weights[k] * prevLayer->neurons[k].output;
            }
            currentLayer->neurons[j].output = sigmoid(activation);
        }
    }
}

void backwardPass(MLP *net, double targets[]) {
    Layer *outputLayer = &net->layers[net->numLayers - 1];
    for (int i = 0; i < outputLayer->numNeurons; i++) {
        double error = targets[i] - outputLayer->neurons[i].output;
        outputLayer->neurons[i].delta = error * sigmoidDerivative(outputLayer->neurons[i].output);
    }

    for (int i = net->numLayers - 2; i >= 0; i--) {
        Layer *hiddenLayer = &net->layers[i];
        Layer *nextLayer = &net->layers[i + 1];
        for (int j = 0; j < hiddenLayer->numNeurons; j++) {
            double error = 0.0;
            for (int k = 0; k < nextLayer->numNeurons; k++) {
                error += nextLayer->neurons[k].weights[j] * nextLayer->neurons[k].delta;
            }
            hiddenLayer->neurons[j].delta = error * sigmoidDerivative(hiddenLayer->neurons[j].output);
        }
    }
}

void updateWeights(MLP *net, double inputs[]) {
    for (int i = 0; i < net->numLayers; i++) {
        double *currentInputs = (i == 0) ? inputs : NULL;
        Layer *prevLayer = (i > 0) ? &net->layers[i-1] : NULL;

        for (int j = 0; j < net->layers[i].numNeurons; j++) {
            for (int k = 0; k < net->layers[i].neurons[j].numInputs; k++) {
                double inputVal = (i == 0) ? currentInputs[k] : prevLayer->neurons[k].output;
                net->layers[i].neurons[j].weights[k] += LEARNING_RATE * net->layers[i].neurons[j].delta * inputVal;
            }
            net->layers[i].neurons[j].bias += LEARNING_RATE * net->layers[i].neurons[j].delta;
        }
    }
}

double* runTrainingSession(const char* sessionTitle, int numArchLayers, const int architecture[],
                           double allData[][4], IrisSpecies allLabels[],
                           IrisSpecies class1, IrisSpecies class2, IrisSpecies class3, int isMultiClass) {
    printf("Starting session: %s\n", sessionTitle);

    double sessionData[TOTAL_SAMPLES][4];
    double sessionTargets[TOTAL_SAMPLES][3] = {{0}};
    int sessionCount = 0;

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        int useSample = 0;
        if (isMultiClass) {
            useSample = 1;
        } else {
            if (allLabels[i] == class1 || allLabels[i] == class2 || allLabels[i] == class3) useSample = 1;
        }

        if (useSample) {
            memcpy(sessionData[sessionCount], allData[i], 4 * sizeof(double));
            if (isMultiClass) {
                sessionTargets[sessionCount][allLabels[i]] = 1.0;
            } else {
                sessionTargets[sessionCount][0] = (allLabels[i] == class1);
            }
            sessionCount++;
        }
    }

    int numTraining = sessionCount * 0.8;
    int numTesting = sessionCount - numTraining;
    
    MLP *net = createMLP(numArchLayers, architecture);
    double *accuracies = (double*)malloc(MAX_EPOCHS * sizeof(double));

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        for (int i = 0; i < numTraining; i++) {
            forwardPass(net, sessionData[i]);
            backwardPass(net, sessionTargets[i]);
            updateWeights(net, sessionData[i]);
        }
        int correct = 0;
        for (int i = numTraining; i < sessionCount; i++) {
            forwardPass(net, sessionData[i]);
            Layer *outputLayer = &net->layers[net->numLayers - 1];
            if (isMultiClass) {
                int predictedClass = 0;
                for(int j = 1; j < outputLayer->numNeurons; j++) {
                    if(outputLayer->neurons[j].output > outputLayer->neurons[predictedClass].output) {
                        predictedClass = j;
                    }
                }
                int trueClass = 0;
                for(int j = 1; j < 3; j++) {
                    if(sessionTargets[i][j] > sessionTargets[i][trueClass]) {
                        trueClass = j;
                    }
                }
                if (predictedClass == trueClass) correct++;
            } else {
                int prediction = outputLayer->neurons[0].output > 0.5;
                if (prediction == (int)sessionTargets[i][0]) correct++;
            }
        }
        accuracies[epoch] = (double)correct / numTesting * 100.0;
    }

    freeMLP(net);
    return accuracies;
}

IrisSpecies speciesNameToEnum(char *species) {
    if (strcmp(species, "Iris-setosa") == 0) return SETOSA;
    if (strcmp(species, "Iris-versicolor") == 0) return VERSICOLOR;
    if (strcmp(species, "Iris-virginica") == 0) return VIRGINICA;
    return -1;
}