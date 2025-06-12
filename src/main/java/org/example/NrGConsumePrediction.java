package org.example;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class NrGConsumePrediction {

    private static final Logger log = LoggerFactory.getLogger(NrGConsumePrediction.class);

    public static void main(String[] args) throws Exception {
        final int batchSize = 1000;
        final int nEpochs = 100;
        int seed = 12345;
        double learningRate = 0.3;
        int numInputs = 4;
        int numOutputs = 1;
        double targetMSE = 0.01;

        SplitTestAndTrain testAndTrain = getSplitTestAndTrain(batchSize);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        MultiLayerConfiguration conf = buildMultiLayerConfiguration(seed, learningRate, numInputs, numOutputs);
        MultiLayerNetwork net = buildMultiLayerNetwork(conf);

        log.debug("Fit training data");
        for (int i = 0; i < nEpochs; i++) {
            long startTime = System.currentTimeMillis();

            net.fit(trainingData);

            long endTime = System.currentTimeMillis();
            long timeTaken = endTime - startTime;

            log.info("Epoch {} completed in {} ms", i + 1, timeTaken);
            logWeights(net);
            RegressionEvaluation eval = new RegressionEvaluation(1);
            INDArray output = net.output(testData.getFeatures(), false);
            eval.eval(testData.getLabels(), output);
            double currentMSE = eval.meanSquaredError(0);
            if (currentMSE <= targetMSE) {
                log.debug("Stopping training as MSE reached the target value.");
                break;
            }
        }
    }

    private static MultiLayerNetwork buildMultiLayerNetwork(MultiLayerConfiguration conf) {
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }

    private static MultiLayerConfiguration buildMultiLayerConfiguration(int seed, double learningRate, int numInputs, int numOutputs) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(2)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(2).nOut(2)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(2).nOut(numOutputs).build())
                .build();
        return conf;
    }

    private static SplitTestAndTrain getSplitTestAndTrain(int batchSize) throws IOException, InterruptedException {

        /**
         * CSV headers
         * 'Month', 'AvgTemp', 'NumRes', 'sqFt', "NrGConsump"
         */

        CSVRecordReader csvRecordReader = new CSVRecordReader(0, '\t');
        FileSplit inputSplit = new FileSplit(new File("src/main/resources/nrgconsump.txt"));
        csvRecordReader.initialize(inputSplit);

        Schema finalSchema = buildSchema();

        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator.Builder(csvRecordReader, batchSize)
                .regression(finalSchema.getIndexOfColumn("NrGconsump"))
                .build();

        DataSet allData = trainIterator.next();

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.70);
        return testAndTrain;
    }



    private static Schema buildSchema() {
        Schema schema = new Schema.Builder()
                .addColumnDouble("Month")
                .addColumnDouble("AvgTemp")
                .addColumnDouble("NumRes")
                .addColumnDouble("SqFt")
                .addColumnDouble("NrGconsump")
                .build();
        return schema;
    }

    private static void logWeights(MultiLayerNetwork net) {
        for (int i = 0; i < net.getnLayers(); i++) {
            log.info("Layer {}: {}", i + 1, net.getLayer(i).getParam("W"));
        }
    }
}