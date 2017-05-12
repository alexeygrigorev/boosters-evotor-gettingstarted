package evotor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.alexeygrigorev.dstools.data.Dataset;

import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import smile.data.Datum;
import smile.data.SparseDataset;
import smile.math.SparseArray;
import smile.math.SparseArray.Entry;

public class XgbUtils {

    public static DMatrix wrapData(Dataset data) throws XGBoostError {
        int nrow = data.length();
        double[][] X = data.getX();
        double[] y = data.getY();

        List<LabeledPoint> points = null;

        if (y != null) {
            points = labeledData(nrow, X, y);
        } else {
            points = unlabeledData(nrow, X);
        }

        String cacheInfo = "";
        return new DMatrix(points.iterator(), cacheInfo);
    }

    private static List<LabeledPoint> labeledData(int nrow, double[][] X, double[] y) {
        List<LabeledPoint> points = new ArrayList<>();

        for (int i = 0; i < nrow; i++) {
            float label = (float) y[i];
            float[] floatRow = asFloat(X[i]);
            LabeledPoint point = LabeledPoint.fromDenseVector(label, floatRow);
            points.add(point);
        }

        return points;
    }

    private static List<LabeledPoint> unlabeledData(int nrow, double[][] X) {
        List<LabeledPoint> points = new ArrayList<>();
        float label = 0.0f;

        for (int i = 0; i < nrow; i++) {
            float[] floatRow = asFloat(X[i]);
            LabeledPoint point = LabeledPoint.fromDenseVector(label, floatRow);
            points.add(point);
        }

        return points;
    }

    public static DMatrix wrapData(SparseDataset data) throws XGBoostError {
        int nrow = data.size();
        List<LabeledPoint> points = new ArrayList<>();

        for (int i = 0; i < nrow; i++) {
            Datum<SparseArray> datum = data.get(i);
            float label = (float) datum.y;
            SparseArray array = datum.x;

            int size = array.size();

            int[] indices = new int[size];
            float[] values = new float[size];

            int idx = 0;
            for (Entry e : array) {
                indices[idx] = e.i;
                values[idx] = (float) e.x;
                idx++;
            }

            LabeledPoint point = LabeledPoint.fromSparseVector(label, indices, values);
            points.add(point);
        }

        String cacheInfo = "";
        return new DMatrix(points.iterator(), cacheInfo);
    }


    public static float[] asFloat(double[] ds) {
        float[] result = new float[ds.length];
        for (int i = 0; i < ds.length; i++) {
            result[i] = (float) ds[i];
        }
        return result;
    }

    public static double[] predictBinary(Booster model, DMatrix data) throws XGBoostError {
        float[][] res = model.predict(data);
        return XgbUtils.unwrapToDouble(res);
    }

    private static double[] unwrapToDouble(float[][] floatResults) {
        int n = floatResults.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = floatResults[i][0];
        }
        return result;
    }

    public static double[] predictBinary(Booster model, Dataset data) throws XGBoostError {
        DMatrix dmatrix = wrapData(data);
        return predictBinary(model, dmatrix);
    }

    public static int[] predictBestClass(Booster model, Dataset data) throws XGBoostError {
        DMatrix dmatrix = wrapData(data);
        return predictBestClass(model, dmatrix);
    }

    public static int[] predictBestClass(Booster model, DMatrix data) throws XGBoostError {
        float[][] resultPredictions = model.predict(data);
        return rowArgMax(resultPredictions);
    }

    private static int[] rowArgMax(float[][] array) {
        int n = array.length;
        int[] result = new int[n];

        for (int i = 0; i < n; i++) {
            result[i] = argmax(array[i]);
        }

        return result;
    }

    private static int argmax(float[] elems) {
        int bestIdx = 0;
        float best = elems[0];

        for (int i = 0; i < elems.length; i++) {
            float elem = elems[i];
            if (elem > best) {
                best = elem;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    public static Map<String, Object> defaultParams() {
        Map<String, Object> params = new HashMap<>();
        params.put("eta", 0.3);
        params.put("gamma", 0);
        params.put("max_depth", 6);
        params.put("min_child_weight", 1);
        params.put("max_delta_step", 0);
        params.put("subsample", 1);
        params.put("colsample_bytree", 1);
        params.put("colsample_bylevel", 1);
        params.put("lambda", 1);
        params.put("alpha", 0);
        params.put("tree_method", "approx");
        params.put("objective", "binary:logistic");
        params.put("eval_metric", "logloss");
        params.put("nthread", 8);
        params.put("seed", 42);
        params.put("silent", 1);
        return params;
    }

}