package com.company;

import java.util.List;

public class Main {

    public static void main(String[] args) {
        List<int[][]> trainingData = Reader.getImages("C:\\Users\\garre\\Development\\NeuralNet\\src\\com\\company\\train-images.idx3-ubyte");
        int[] trainingLables = Reader.getLabels("C:\\Users\\garre\\Development\\NeuralNet\\src\\com\\company\\train-labels.idx1-ubyte");
        List<int[][]> testingData = Reader.getImages("C:\\Users\\garre\\Development\\NeuralNet\\src\\com\\company\\t10k-images.idx3-ubyte");
        int[] testingLables = Reader.getLabels("C:\\Users\\garre\\Development\\NeuralNet\\src\\com\\company\\t10k-labels.idx1-ubyte");

        Network nn = new Network(784, 30, 10);
        int[] in = Network.prepareImageInput(trainingData.get(1));
        //System.out.println(nn.feedForward(in));
        System.out.println(nn.evaluate(testingData, testingLables));
        nn.SGD(trainingData, trainingLables, 10, 10, 3, testingData, testingLables);
    }
}
