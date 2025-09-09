#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_INPUTS 4
#define MAX_EPOCHS 100
#define TOTAL_SAMPLES 150

// Enum for species to make code more readable
typedef enum {
    SETOSA,
    VERSICOLOR,
    VIRGINICA
} IrisSpecies;

// Perceptron structure using camelCase
typedef struct {
    double *weights;
    double bias;
    double learningRate;
} Perceptron;

// Function prototypes in camelCase
void initializePerceptron(Perceptron *p);
void freePerceptron(Perceptron *p);
double relu(double x);
double reluDerivative(double x);
double forwardPass(Perceptron *p, double *inputs);
int predict(Perceptron *p, double *inputs);
void train(Perceptron *p, double *inputs, int desiredOutput);
IrisSpecies speciesNameToEnum(char *species);
void runTrainingSession(FILE *outputFile, const char* sessionTitle, 
                          double allData[][NUM_INPUTS], IrisSpecies allLabels[], 
                          IrisSpecies positiveClass, IrisSpecies negativeClass1, IrisSpecies negativeClass2);

int main() {
    srand(time(NULL));

    // --- 1. Load Entire Dataset from File ---
    double allData[TOTAL_SAMPLES][NUM_INPUTS];
    IrisSpecies allLabels[TOTAL_SAMPLES];

    FILE *file = fopen("iris.data", "r");
    if (file == NULL) {
        printf("Error: Could not open iris.data.\n");
        return 1;
    }

    char line[100];
    int sampleCount = 0;
    while (fgets(line, sizeof(line), file) && sampleCount < TOTAL_SAMPLES) {
        char speciesName[20];
        sscanf(line, "%lf,%lf,%lf,%lf,%s", 
               &allData[sampleCount][0], &allData[sampleCount][1], 
               &allData[sampleCount][2], &allData[sampleCount][3], 
               speciesName);
        allLabels[sampleCount] = speciesNameToEnum(speciesName);
        sampleCount++;
    }
    fclose(file);

    // --- 2. Open Output File and Run All Training Sessions ---
    FILE *resultsFile = fopen("results_ReLU.txt", "w");
    if (resultsFile == NULL) {
        printf("Error: Could not create results.txt for writing.\n");
        return 1;
    }

    printf("Running all 6 training sessions with ReLU... This may take a moment.\n");

    // --- Part 1: One-vs-Rest Training ---
    runTrainingSession(resultsFile, "Train 1: Setosa vs. Rest", allData, allLabels, SETOSA, VERSICOLOR, VIRGINICA);
    runTrainingSession(resultsFile, "Train 2: Versicolor vs. Rest", allData, allLabels, VERSICOLOR, SETOSA, VIRGINICA);
    runTrainingSession(resultsFile, "Train 3: Virginica vs. Rest", allData, allLabels, VIRGINICA, SETOSA, VIRGINICA);

    // --- Part 2: One-vs-One (Pair-to-Pair) Training ---
    runTrainingSession(resultsFile, "Train 4: Setosa vs. Versicolor", allData, allLabels, SETOSA, VERSICOLOR, -1);
    runTrainingSession(resultsFile, "Train 5: Setosa vs. Virginica", allData, allLabels, SETOSA, VIRGINICA, -1);
    runTrainingSession(resultsFile, "Train 6: Versicolor vs. Virginica", allData, allLabels, VERSICOLOR, VIRGINICA, -1);

    fclose(resultsFile);
    printf("Training complete. Results exported to results.txt\n");

    return 0;
}

/**
 * @brief Runs a full training and testing session for a given classification scenario.
 */
void runTrainingSession(FILE *outputFile, const char* sessionTitle, 
                          double allData[][NUM_INPUTS], IrisSpecies allLabels[], 
                          IrisSpecies positiveClass, IrisSpecies negativeClass1, IrisSpecies negativeClass2) {
    
    printf("Starting session: %s\n", sessionTitle);

    double sessionData[TOTAL_SAMPLES][NUM_INPUTS];
    int sessionLabels[TOTAL_SAMPLES];
    int sessionSampleCount = 0;

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        if (allLabels[i] == positiveClass) {
            memcpy(sessionData[sessionSampleCount], allData[i], NUM_INPUTS * sizeof(double));
            sessionLabels[sessionSampleCount] = 1;
            sessionSampleCount++;
        } else if (allLabels[i] == negativeClass1 || allLabels[i] == negativeClass2) {
            memcpy(sessionData[sessionSampleCount], allData[i], NUM_INPUTS * sizeof(double));
            sessionLabels[sessionSampleCount] = 0;
            sessionSampleCount++;
        }
    }

    int numTrainingSamples = sessionSampleCount * 0.8;
    int numTestingSamples = sessionSampleCount - numTrainingSamples;

    double trainingData[numTrainingSamples][NUM_INPUTS];
    int trainingOutputs[numTrainingSamples];
    double testingData[numTestingSamples][NUM_INPUTS];
    int testingOutputs[numTestingSamples];
    
    int trainIndex = 0, testIndex = 0;
    for (int i = 0; i < sessionSampleCount; i++) {
        if (i % 5 < 4) {
            if (trainIndex < numTrainingSamples) {
                memcpy(trainingData[trainIndex], sessionData[i], NUM_INPUTS * sizeof(double));
                trainingOutputs[trainIndex++] = sessionLabels[i];
            }
        } else {
             if (testIndex < numTestingSamples) {
                memcpy(testingData[testIndex], sessionData[i], NUM_INPUTS * sizeof(double));
                testingOutputs[testIndex++] = sessionLabels[i];
            }
        }
    }

    Perceptron p;
    initializePerceptron(&p);
    
    double bestAccuracy = -1.0;
    double bestWeights[NUM_INPUTS];
    double bestBias = 0.0;
    double epochAccuracies[MAX_EPOCHS];

    for (int i = 0; i < MAX_EPOCHS; i++) {
        for (int j = 0; j < numTrainingSamples; j++) {
            train(&p, trainingData[j], trainingOutputs[j]);
        }
        
        int correctPredictions = 0;
        for (int j = 0; j < numTestingSamples; j++) {
            if (predict(&p, testingData[j]) == testingOutputs[j]) {
                correctPredictions++;
            }
        }
        double currentAccuracy = (double)correctPredictions / numTestingSamples * 100.0;
        epochAccuracies[i] = currentAccuracy;

        if (currentAccuracy >= bestAccuracy) {
            bestAccuracy = currentAccuracy;
            memcpy(bestWeights, p.weights, NUM_INPUTS * sizeof(double));
            bestBias = p.bias;
        }
    }

    fprintf(outputFile, "%s\n", sessionTitle);
    fprintf(outputFile, "Best Weights: [%.4f, %.4f, %.4f, %.4f], Bias: %.4f (Achieved %.2f%% Accuracy)\n", 
            bestWeights[0], bestWeights[1], bestWeights[2], bestWeights[3], bestBias, bestAccuracy);
    
    fprintf(outputFile, "Precision through epochs:\n");
    for (int i = 0; i < MAX_EPOCHS; i++) {
        fprintf(outputFile, "epoch %d: %.2f%%\n", i + 1, epochAccuracies[i]);
    }
    fprintf(outputFile, "\n\n");

    freePerceptron(&p);
}

// --- Helper and Activation Functions ---

IrisSpecies speciesNameToEnum(char *species) {
    if (strcmp(species, "Iris-setosa") == 0) return SETOSA;
    if (strcmp(species, "Iris-versicolor") == 0) return VERSICOLOR;
    if (strcmp(species, "Iris-virginica") == 0) return VIRGINICA;
    return -1;
}

void initializePerceptron(Perceptron *p) {
    p->weights = (double*)malloc(NUM_INPUTS * sizeof(double));
    p->learningRate = 0.01;
    for (int i = 0; i < NUM_INPUTS; i++) {
        p->weights[i] = ((double)rand() / RAND_MAX) * 0.1;
    }
    p->bias = ((double)rand() / RAND_MAX) * 0.1;
}

void freePerceptron(Perceptron *p) {
    free(p->weights);
    p->weights = NULL;
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double reluDerivative(double x) {
    return x > 0 ? 1 : 0;
}

// --- Core Neuron Logic ---

double forwardPass(Perceptron *p, double *inputs) {
    double weightedSum = p->bias;
    for (int i = 0; i < NUM_INPUTS; i++) {
        weightedSum += p->weights[i] * inputs[i];
    }
    return relu(weightedSum);
}

int predict(Perceptron *p, double *inputs) {
    return forwardPass(p, inputs) > 0.5 ? 1 : 0;
}

void train(Perceptron *p, double *inputs, int desiredOutput) {
    double weightedSum = p->bias;
    for (int i = 0; i < NUM_INPUTS; i++) {
        weightedSum += p->weights[i] * inputs[i];
    }
    double predictedOutput = relu(weightedSum);

    double error = predictedOutput - desiredOutput;
    double derivative = reluDerivative(weightedSum);

    for (int i = 0; i < NUM_INPUTS; i++) {
        double gradient = error * derivative * inputs[i];
        p->weights[i] -= p->learningRate * gradient;
    }
    p->bias -= p->learningRate * error * derivative;
}