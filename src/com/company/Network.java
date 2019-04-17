package com.company;

import java.util.ArrayList;
import java.util.List;

public class Network {
    private int layers;
    private int[] properties;
    private Matrix[] biases;
    private Matrix[] weights;

    public Network(int ... properties){
        this.properties = properties;
        this.layers = properties.length;
        this.biases = new Matrix[layers - 1];
        this.weights = new Matrix[layers - 1];
        for(int i = 0; i < layers - 1; i++){
            weights[i] = new Matrix(properties[i], properties[i + 1]);
            biases[i] = new Matrix(1, properties[i + 1]);
            weights[i].randomize(-1, 1, .01);
            biases[i].randomize(-1, 1, .01);
        }
    }

    public Matrix feedForward(int[] a){
        Matrix input = prepareInput(a);
        Matrix output = input;
        if(a.length != properties[0]){
            System.out.println(a.length);
            System.out.println(properties[0]);
            throw new RuntimeException("Invalid Input");
        }
        for(int i = 0; i < layers - 1; i++) {
            output = weights[i].multiply(input).add(biases[i]);
            input = output.elementWise(Network::sigmoid);
        }
        return input;
    }

    public Matrix prepareInput(int[] in){
        double[][] newArr = new double[in.length][1];
        for(int i = 0; i < in.length; i++){
            newArr[i][0] = in[i];
        }
        return new Matrix(newArr);
    }


    public void SGD(List<int[][]> trainingData, int[] lables, int iterations, int batchSize, double learningRate, List<int[][]> testingData, int[] testingLables){
        List<int[]> in = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        // Convert our data to a 1D array (our input nodes)
        for (int[][] t : trainingData) {
            in.add(prepareImageInput(t));
        }
        if(testingData != null) {
            for (int[][] t : testingData) {
                test.add(prepareImageInput(t));
            }
        }

        for(int i = 0; i < iterations; i++){
            int batches = (in.size() / batchSize);
            for(int j = 0; j < batches; j++){
                updateBatch(
                        in.subList(j * batchSize, (j + 1) * batchSize),
                        subArr(lables, j * batchSize, (j + 1) * batchSize),
                        learningRate
                );
            }
            System.out.println("Epoch" + (i + 1) + "Completed");
            System.out.println(evaluate(testingData, testingLables));
        }
    }

    public void updateBatch(List<int[]> batch, int[] target, double learningRate){
        List<Matrix> nabla_b = new ArrayList<>();
        List<Matrix> nabla_w = new ArrayList<>();
        for(Matrix b : biases)  nabla_b.add(b.zero());
        for(Matrix w : weights) nabla_w.add(w.zero());
        for(int i = 0; i < batch.size(); i++){
            List<Matrix>[] backProp = backPropogate(batch.get(i), target[i]);
            List<Matrix> deltaNablaB = backProp[0];
            List<Matrix> deltaNablaW = backProp[1];
            for(int j = 0; j < deltaNablaB.size(); j++){
                nabla_b.set(j, nabla_b.get(j).add(deltaNablaB.get(j)));
            }for(int j = 0; j < deltaNablaW.size(); j++){
//                System.out.println(j);
//                System.out.println(nabla_w.get(j));
//                System.out.println(deltaNablaW.get(j));
                //System.out.println(deltaNablaW.get(1));
                nabla_w.set(j, nabla_w.get(j).add(deltaNablaW.get(j)));
            }
        }
        //System.out.println(nabla_w.get(1));
        // Apply Changes
        for(int i = 0; i < weights.length; i++){
            weights[i] = weights[i].add(nabla_w.get(i).elementWise((x) -> x * (learningRate/batch.size()))); //elementWiseAdd(-learningRate / batch.size()).elementWiseMultiply(nabla_w.get(i));
        }
        for(int i = 0; i < biases.length; i++){
            biases[i] = biases[i].add(nabla_b.get(i).elementWise((x)-> x * (learningRate/batch.size())));//elementWiseAdd(-learningRate / batch.size()).elementWiseMultiply(nabla_b.get(i));
        }
    }

    public void printWeights(){
        for(int i = 0; i < weights.length; i++){
            System.out.println(weights[i]);
        }
        for(int i = 0; i < biases.length; i++){
            System.out.println(biases[i]);
        }
    }

    @SuppressWarnings("Unchecked")
    public List<Matrix>[] backPropogate(int[] input, int target){
        List<Matrix> nabla_b = new ArrayList<>();
        List<Matrix> nabla_w = new ArrayList<>();
        for(Matrix b : biases)  nabla_b.add(b.zero());
        for(Matrix w : weights) nabla_w.add(w.zero());
        Matrix activation = prepareInput(input);
        Matrix y = prepareTargetInput(target);
        List<Matrix> activations = new ArrayList<>();
        activations.add(activation);
        List<Matrix> zs = new ArrayList<>();
        for(int i = 0; i < layers - 1; i++){
            Matrix z = weights[i].multiply(activation).add(biases[i]);
//            System.out.println("Layer: " + i);
//            System.out.println(weights[i]);
//            System.out.println(activation.transpose());
//            System.out.println(biases[i]);
//            System.out.println(z);
            zs.add(z);
            activation = z.elementWise(Network::sigmoid);
            activations.add(activation);
        }
        int len = activations.size();

        // Calculate backPropagating error zs.get(zs.size() - 1)
        Matrix delta = y.subtract(activations.get(len - 1)).elementWiseMultiply(zs.get(zs.size() - 1).elementWise(Network::sigmoidPrime)).elementWise(x -> -x);
        //System.out.println(delta);
        // Store changes
        nabla_b.set(nabla_b.size() - 1, delta);
        nabla_w.set(nabla_w.size() - 1, delta.multiply(activations.get(len - 2)).transpose());
        // Continue Propagating
        for(int l = layers - 2; l >= 1; l--){
            //System.out.println(delta);
            Matrix z = zs.get(l - 1);
            Matrix sp = z.elementWise(Network::sigmoidPrime);
            delta = weights[l].transpose().multiply(delta).elementWiseMultiply(sp);
//            System.out.println("Activity");
//            System.out.println(z);
//            System.out.println("Prime Activity");
//            System.out.println(sp);
            nabla_b.set(l, delta);
            // System.out.println(delta.multiply(activations.get(l - 1).transpose()));
            // System.out.println(delta);
            // System.out.println(activations.get(l - 1));
            System.out.println(nabla_w.size());
            nabla_w.set(l, delta.multiply(activations.get(l + 1).transpose()));
        }

        return new List[]{nabla_b, nabla_w};
    }
//    public Matrix costDerivative(Matrix activation){
//        return activation.;
//    }

    public static double sigmoidPrime(double a){
        return sigmoid(a)*(1 - sigmoid(a));
    }

    public int[] subArr(int[] arr, int from, int to){
        int[] newArr = new int[to - from];
        for(int i = from; i < to; i++){
            newArr[i - from] = arr[i];
        }
        return newArr;
    }

    public static int[] prepareImageInput(int[][] in){
        int[] newArr = new int[in.length * in[0].length];
        for(int i = 0; i < in.length; i++){
            for(int j = 0; j < in[i].length; j++){
                newArr[i * in.length + j] = in[i][j];
            }
        }
        return newArr;
    }


    public Matrix prepareTargetInput(int n){
        double[][] newArr = new double[10][1];
        newArr[n][0] = 1;
        return new Matrix(newArr);
    }

    public static double sigmoid(double a){
        return 1.0 / (1.0 + Math.exp(-a));
    }

    public int evaluate(List<int[][]> testData, int[] testLables){
        int count =  0;
        for(int i = 0; i < testData.size(); i++){
            Matrix out = feedForward(prepareImageInput(testData.get(i)));
            if(out.max() == testLables[i]) {
                count++;
            }
        }
        return count;
    }
}
