#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_INPUTS 4
#define TOTAL_SAMPLES 150
#define NUM_TRIALS 10 // Number of times to run each combination to average results

// Enum and Perceptron struct
typedef enum { SETOSA, VERSICOLOR, VIRGINICA } IrisSpecies;
typedef struct {
    double *weights;
    double bias;
    double learningRate;
} Perceptron;

// A struct to hold the results of a single tuning run
typedef struct {
    double learningRate;
    int epochs;
    double accuracy;
} TrialResult;

// Function prototypes
void initializePerceptron(Perceptron *p, double learningRate);
void freePerceptron(Perceptron *p);
int predict(Perceptron *p, double *inputs);
void train(Perceptron *p, double *inputs, int desiredOutput);
IrisSpecies speciesNameToEnum(char *species);
double runSingleTrial(int numEpochs, double learningRate, int numTrainingSamples, double trainingData[][NUM_INPUTS], int trainingOutputs[], int numTestingSamples, double testingData[][NUM_INPUTS], int testingOutputs[]);

int main() {
    srand(time(NULL));

    // --- 1. Define the Hyperparameter Grid (7x7 = 49 combinations) ---
    double learningRates[] = {0.5, 0.2, 0.1, 0.08, 0.05, 0.02, 0.01};
    int epochOptions[] = {20, 50, 100, 150, 200, 300, 500};
    int numLearningRates = sizeof(learningRates) / sizeof(learningRates[0]);
    int numEpochOptions = sizeof(epochOptions) / sizeof(epochOptions[0]);

    // --- 2. Load and Prepare Data for "Versicolor vs. Virginica" ---
    double allData[TOTAL_SAMPLES][NUM_INPUTS];
    IrisSpecies allLabels[TOTAL_SAMPLES];

    FILE *file = fopen("iris.data", "r");
    if (file == NULL) { 
        printf("Error: Could not open iris.data. Make sure it is in the same folder.\n"); 
        return 1; 
    }
    
    char line[100];
    int count = 0;
    while (fgets(line, sizeof(line), file) && count < TOTAL_SAMPLES) {
        char name[20];
        sscanf(line, "%lf,%lf,%lf,%lf,%s", &allData[count][0], &allData[count][1], &allData[count][2], &allData[count][3], name);
        allLabels[count] = speciesNameToEnum(name);
        count++;
    }
    fclose(file);

    // Filter for only Versicolor and Virginica
    double sessionData[100][NUM_INPUTS];
    int sessionLabels[100];
    int sessionCount = 0;
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        if (allLabels[i] == VERSICOLOR || allLabels[i] == VIRGINICA) {
            memcpy(sessionData[sessionCount], allData[i], NUM_INPUTS * sizeof(double));
            sessionLabels[sessionCount] = (allLabels[i] == VIRGINICA); // Versicolor=0, Virginica=1
            sessionCount++;
        }
    }

    // Split data into train/test sets (80% train, 20% test)
    int numTrainingSamples = 80;
    int numTestingSamples = 20;
    double trainingData[numTrainingSamples][NUM_INPUTS];
    int trainingOutputs[numTrainingSamples];
    double testingData[numTestingSamples][NUM_INPUTS];
    int testingOutputs[numTestingSamples];

    int trainIndex = 0, testIndex = 0;
    for (int i = 0; i < sessionCount; i++) {
        if (i % 5 < 4) { 
            memcpy(trainingData[trainIndex], sessionData[i], NUM_INPUTS * sizeof(double)); 
            trainingOutputs[trainIndex++] = sessionLabels[i]; 
        } else { 
            memcpy(testingData[testIndex], sessionData[i], NUM_INPUTS * sizeof(double)); 
            testingOutputs[testIndex++] = sessionLabels[i]; 
        }
    }
    
    // --- 3. Run the Grid Search ---
    int numCombinations = numLearningRates * numEpochOptions;
    TrialResult* allResults = (TrialResult*)malloc(numCombinations * NUM_TRIALS * sizeof(TrialResult));
    int resultIndex = 0;

    double bestAvgAccuracy = -1.0;
    double bestLearningRate = 0;
    int bestEpochs = 0;

    printf("Starting hyperparameter tuning for Versicolor vs. Virginica...\n");
    printf("Total combinations: %d. Trials per combination: %d. Total runs: %d\n\n", numCombinations, NUM_TRIALS, numCombinations * NUM_TRIALS);

    for (int i = 0; i < numLearningRates; i++) {
        for (int j = 0; j < numEpochOptions; j++) {
            double currentLearningRate = learningRates[i];
            int currentEpochs = epochOptions[j];
            double totalAccuracy = 0;

            for (int k = 0; k < NUM_TRIALS; k++) {
                double accuracy = runSingleTrial(currentEpochs, currentLearningRate, numTrainingSamples, trainingData, trainingOutputs, numTestingSamples, testingData, testingOutputs);
                allResults[resultIndex].learningRate = currentLearningRate;
                allResults[resultIndex].epochs = currentEpochs;
                allResults[resultIndex].accuracy = accuracy;
                resultIndex++;
                totalAccuracy += accuracy;
            }

            double avgAccuracy = totalAccuracy / NUM_TRIALS;
            printf("LR: %.2f, Epochs: %d -> Average Accuracy: %.2f%%\n", currentLearningRate, currentEpochs, avgAccuracy);

            if (avgAccuracy > bestAvgAccuracy) {
                bestAvgAccuracy = avgAccuracy;
                bestLearningRate = currentLearningRate;
                bestEpochs = currentEpochs;
            }
        }
    }

    // --- 4. Report and Export Results ---
    printf("\n--- Tuning Complete ---\n");
    printf("Best Hyperparameters Found:\n");
    printf("  - Learning Rate: %.2f\n", bestLearningRate);
    printf("  - Epochs: %d\n", bestEpochs);
    printf("  - Best Average Accuracy: %.2f%%\n", bestAvgAccuracy);

    FILE* resultsFile = fopen("hyperparameter_results.csv", "w");
    if (resultsFile) {
        fprintf(resultsFile, "LearningRate,Epochs,Accuracy\n");
        for (int i = 0; i < resultIndex; i++) {
            fprintf(resultsFile, "%.2f,%d,%.2f\n", allResults[i].learningRate, allResults[i].epochs, allResults[i].accuracy);
        }
        fclose(resultsFile);
        printf("\nDetailed trial results exported to hyperparameter_results.csv\n");
    }

    free(allResults);
    return 0;
}

/**
 * @brief This function runs a single, complete training/testing cycle and returns the accuracy.
 */
double runSingleTrial(int numEpochs, double learningRate, int numTrainingSamples, double trainingData[][NUM_INPUTS], int trainingOutputs[], int numTestingSamples, double testingData[][NUM_INPUTS], int testingOutputs[]) {
    Perceptron p;
    initializePerceptron(&p, learningRate);

    for (int i = 0; i < numEpochs; i++) {
        for (int j = 0; j < numTrainingSamples; j++) {
            train(&p, trainingData[j], trainingOutputs[j]);
        }
    }

    int correct = 0;
    for (int i = 0; i < numTestingSamples; i++) {
        if (predict(&p, testingData[i]) == testingOutputs[i]) {
            correct++;
        }
    }

    freePerceptron(&p);
    return (double)correct / numTestingSamples * 100.0;
}

// --- Standard Helper and Perceptron Functions ---

IrisSpecies speciesNameToEnum(char *species) {
    if (strcmp(species, "Iris-setosa") == 0) return SETOSA;
    if (strcmp(species, "Iris-versicolor") == 0) return VERSICOLOR;
    if (strcmp(species, "Iris-virginica") == 0) return VIRGINICA;
    return -1;
}

void initializePerceptron(Perceptron *p, double learningRate) {
    p->weights = (double*)malloc(NUM_INPUTS * sizeof(double));
    p->learningRate = learningRate; // Use the passed-in hyperparameter
    for (int i = 0; i < NUM_INPUTS; i++) { p->weights[i] = ((double)rand() / RAND_MAX) - 0.5; }
    p->bias = ((double)rand() / RAND_MAX) - 0.5;
}

void freePerceptron(Perceptron *p) { 
    free(p->weights); 
    p->weights = NULL; 
}

int predict(Perceptron *p, double *inputs) {
    double sum = p->bias;
    for (int i = 0; i < NUM_INPUTS; i++) { sum += p->weights[i] * inputs[i]; }
    return sum >= 0 ? 1 : 0; // Step Function
}

void train(Perceptron *p, double *inputs, int desired) {
    int prediction = predict(p, inputs);
    if (prediction != desired) {
        int error = desired - prediction;
        for (int i = 0; i < NUM_INPUTS; i++) { p->weights[i] += p->learningRate * error * inputs[i]; }
        p->bias += p->learningRate * error;
    }
}