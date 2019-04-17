package com.company;

import java.util.function.Function;

public class Matrix {
    private double[][] data;
    private int length;
    private int height;

    public Matrix(int length, int height){
        data = new double[height][length];
        this.height = height;
        this.length = length;
    }

    public Matrix(double[][] data){
        this.height = data.length;
        this.length = data[0].length;
        this.data = data;
    }

    public void randomize(int low, int high){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < length; j++){
                data[i][j] = ((high - low) * Math.random()) + low;
            }
        }
    }

    public Matrix multiply(Matrix m){
        double[][] newData = new double[height][m.length];
        for(int i = 0; i < newData.length; i++){
            for(int j = 0; j < newData[i].length; j++){
                double sum = 0;
                for(int k = 0; k < data[i].length; k++){
                    sum += data[i][k] * m.data[k][j];
                }
                newData[i][j] = sum;
            }
        }
        return new Matrix(newData);
    }

    public Matrix elementWise(Function<Double, Double> function){
        double[][] newData = new double[height][length];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < length; j++) {
                newData[i][j] = function.apply(data[i][j]);
            }
        }
        return new Matrix(newData);
    }

    public Matrix add(Matrix m){
        double[][] newData = new double[height][length];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < length; j++) {
                newData[i][j] = data[i][j] + m.data[i][j];
            }
        }
        return new Matrix(newData);
    }

    public Matrix subtract(Matrix m){
        double[][] newData = new double[height][length];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < length; j++) {
                newData[i][j] = data[i][j] - m.data[i][j];
            }
        }
        return new Matrix(newData);
    }

    public Matrix transpose(){
        double[][] newData = new double[length][height];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < length; j++) {
                newData[j][i] = data[i][j];
            }
        }
        return new Matrix(newData);
    }

    public Matrix elementWiseMultiply(Matrix m){
        double[][] newData = new double[height][length];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < length; j++) {
                newData[i][j] = data[i][j] * m.data[i][j];
            }
        }
        return new Matrix(newData);
    }

    public Matrix elementWiseAdd(double n){
        double[][] newData = new double[height][length];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < length; j++) {
                newData[i][j] = data[i][j] + n;
            }
        }
        return new Matrix(newData);
    }

    public Matrix zero(){
        Matrix m = new Matrix(length, height);
        return m;
    }



    @Override
    public String toString() {
        String s = "";
        for (double[] row : data) {
            s += "[";
            for(double d : row){
                s += d + " ";
            }
            s += "]\n";
        }
        return s;
    }

    public int max(){
        int currMaxi = 0;
        int currMaxj = 0;
        for(int i = 0; i < height; i++){
            for(int j = 0; j < length; j++){
                if(data[i][j] > data[currMaxi][currMaxj]){
                    currMaxi = i;
                    currMaxj = j;
                }
            }
        }
        return currMaxi;
    }
}
