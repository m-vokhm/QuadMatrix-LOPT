/*

 Copyright 2021-2023 M.Vokhmentsev

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

*/

package com.mvohm.quadmatrix;

import com.mvohm.quadmatrix.Matrix.ErrorCodes;
import com.mvohm.quadmatrix.api.ClientDelegate;
import com.mvohm.quadruple.Quadruple;

import static com.mvohm.quadmatrix.Util.*;

class DoubleMatrixSolver extends MatrixSolver {

  // TODO_for_next_release 2023-05-27 11:10:12 Reorder methods accordingly with their usage

  enum SolutionMethod {NONE, CHOLESKY, LU};

  private final int size;
  private final double[][] matrix;  // Original matrix
  private double[] vector;          // Original vector, used for refinement

  private SolutionMethod lastSolvedWith = SolutionMethod.NONE;
  private ErrorCodes errorCode = ErrorCodes.OK;

  private double[][] luDecomposition;
  private boolean luDecompositionError;

  private double[][] choleskyDecomposition;
  private boolean choleskyDecompositionError;

  private int[] pivot;              // filled by decomposeLU(), used by solveLU()
  private double[] rowScales;
  private double[] solution;

  private double[][] inversion = null;

  private int detSignCorrection = 1;            // To correct the determinant sign in case of swapping rows while pivoting
  private double determinant = Double.NaN;      // Unless already computed
  private Double norm;

  private static ClientDelegate clientDelegate = ClientDelegate.NO_OP;

  DoubleMatrixSolver(double[][] matrix, boolean needToScale) {
    this.size = matrix.length;
    this.matrix = matrix;
    this.needToScale = needToScale;
  }

  static void setClentDelegate(ClientDelegate delegate) {
    clientDelegate = delegate;
  }

  private static final double LU_VECTOR_SOLUTION_OPS_PER_M  = 0.761;    // Approx. elementary ops for 100x100 matrix (N^3 = 1,000,000)
  private static final double CH_VECTOR_SOLUTION_OPS_PER_M  = 0.173;   // Approx. elementary ops for 100x100 matrix (N^3 = 1,000,000)
  private static final double MATRIX_SOLUTION_OPS_PER_M     = 1.74;    // Approx. elementary ops for 100x100 matrix (N^3 = 1,000,000)
  private static final double INVERSION_OPS_PER_M           = 1.74;    // Approx. elementary ops for 100x100 matrix (N^3 = 1,000,000)
  private static final double MULTIPLICATION_OPS_PER_M      = 5.0;    // Approx. elementary ops for 100x100 matrix (N^3 = 1,000,000)

  private static final String LU_VECTOR_SOLUTION_NAME   = "LU vector solution"; // Operation name for a start message
  private static final String CH_VECTOR_SOLUTION_NAME   = "Ch vector solution";
  private static final String MATRIX_SOLUTION_NAME      = "Matrix solution";
  private static final String INVERSION_NAME            = "Inversion";
  private static final String MULTIPLICATION_NAME       = "Multiplication";

  // Add additional constants for other operations as needed.

  private static final double PERCENTAGE_STEP = 5.0;  // Report progress after every 5%

  private long scaleOpCount;


  /* *******************************************************************************
  /***** Solutions with respect of vectors *****************************************
  /*********************************************************************************/

  /**
   * Solves a system formed by the matrix and the given vector
   * using LU-decomposition
   * @param vector
   * @return the solution
   */
  double[] solveLU(double[] vector) {
    this.vector = vector.clone();
    solution = solveLUInternally(vector);
    lastSolvedWith = SolutionMethod.LU;
    return solution.clone();                  // To protect the internally stored solutions for possible subsequent retrievals
  }

  double[] solveLUAccurately(double[] vector) {
    this.vector = vector.clone();
    solution = solveLUInternally(vector);
    lastSolvedWith = SolutionMethod.LU;
    refine();
    return solution.clone();
  }

  double[] solveCholesky(double[] vector) {
    this.vector = vector.clone();
    solution = solveCholeskyInternally(vector);
    lastSolvedWith = SolutionMethod.CHOLESKY;
    return solution.clone();                  // To protect the internally stored solutions for possible subsequent retrievals
  }

  double[] solveCholeskyAccurately(double[] vector) {
    this.vector = vector.clone();
    solution = solveCholeskyInternally(vector);
    lastSolvedWith = SolutionMethod.CHOLESKY;
    refine();
    return solution.clone();
  }

  double[] getSolution() {
    return solution;
  }

  String errorCode() {
    return errorCode.toString();
  }

  /* ********************************************************************************
  /*** Solutions with respect of matrices *******************************************
  /* ********************************************************************************/

  /** Attention -- spoils the matrixB, but returns a newly-created array */
  double[][] solve(double[][] matrixB) {
    clientDelegate.sayStarting(MATRIX_SOLUTION_NAME);
    final double[][] result = doSolve(matrixB);
    clientDelegate.sayDone();
    return result;
  }

  double[][] doSolve(double[][] matrixB) {
    if (luDecompositionError)
      throwNonInvertibleError();

    long opCount = 0;
    if (size > 10) {
      initProgressTracking(MATRIX_SOLUTION_OPS_PER_M, size);
    }

    if (luDecomposition == null) { // The very first invocation.
      opCount = scaleAndDecompose();
    }

    // Copy right hand side with pivoting and scaling
    final double[][] matrixX = scaleAndPermute(matrixB, pivot); // Spoils matrixB, and returns a new array

    // Solve L*Y = B(piv,:)
    for (int k = 0; k < size; k++) {
      for (int i = k + 1; i < size; i++) {
        for (int j = 0; j < size; j++) {
          matrixX[i][j] -= matrixX[k][j] * luDecomposition[i][k];
        }
      }
      opCount += (size - k - 1) * size;
      if (opCount > nextOpsThreshold) {
        updateProgress(opCount, size);
      }
    }

    // Solve U * X = Y;
    for (int k = size - 1; k >= 0; k--) {
      for (int j = 0; j < size; j++) {
        matrixX[k][j] /= luDecomposition[k][k];
      }
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < size; j++) {
          matrixX[i][j] -= matrixX[k][j] * luDecomposition[i][k];
        }
      }
      opCount += size * size;
      if (opCount > nextOpsThreshold) {
        updateProgress(opCount, size);
      }
    }

    return matrixX;
  }

  /* ********************************************************************************
  /*** Inverse **********************************************************************
  /* ********************************************************************************/

  double[][] inverse() {
    if (inversion == null) {
      clientDelegate.sayStarting(INVERSION_NAME);
      inversion = doSolve(unityMatrix());
      clientDelegate.sayDone();
    }
    return inversion;
  }

  double[][] inverseAccurately() {
    return solveAccurately(unityMatrix());
  }

  /**
   * Creates a new two-dimentional double array and fills it with the transposition of the source array
   * @param matrix
   * @return
   */
  double[][] transpose(double[][] matrix) {
    final int size = matrix.length;
    final double[][] result = new double[size][size];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        result[i][j] = matrix[j][i];
      }
    }
    return result;
  }

  double[][] unityMatrix() {
    final double[][] result = new double[size][size];
    for (int i = 0; i < size; i++) {
      result[i][i] = 1;
    }
    return result;
  }

  /* *******************************************************************************
  /* **** Multiplications **********************************************************
  /*********************************************************************************/

  double[][] multiply(double[][] factor) {

    long opCount = 0;
    if (size > 10) {
      clientDelegate.sayStarting(MULTIPLICATION_NAME);
      initProgressTracking(MULTIPLICATION_OPS_PER_M, size);
    }

    factor = transpose(factor);

    final double[][] product = new double[size][size];
    final double[] prodVector = new double[size];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        for (int k = 0; k < size; k++) {
          prodVector[k] = matrix[i][k] * factor[j][k];
        }
        product[i][j] = sumOfVector(prodVector);
        opCount += sumOpCount + size;
        if (opCount > nextOpsThreshold) {
          updateProgress(opCount, size);
        }
      }
    }

    clientDelegate.sayDone();
    return product;
  }

  double[] multiply(double[] factor) {
    final int size = factor.length;
    final double[] result = new double[size];
    for (int i = 0; i < size; i++) {
      final double[] rowProduct = multiplyElements(matrix[i], factor);
      result[i] = sumOfVector(rowProduct);
    }
    return result;
  }

  Quadruple[][] multiply(Quadruple scalar) {
    final Quadruple[][] product = new Quadruple[size][size];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        product[i][j] = Quadruple.multiply(scalar, matrix[i][j]);
      }
    }
    return product;
  }

  /* *******************************************************************************
  /* **** Additions and subtractions ***********************************************
  /*********************************************************************************/

  static double[][] addMatrices(double[][] summandA, double[][] summandB) {
    final int size = summandA.length;
    final double[][] sum = new double[size][size];
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++) {
        sum[i][j] = summandA[i][j] + summandB[i][j];
      }
    return sum;
  }

  static double[][] subtractMatrices(double[][] minuend, double[][] subtrahend) {
    final int size = minuend.length;
    final double[][] difference = new double[size][size];
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++) {
        difference[i][j] = minuend[i][j] - subtrahend[i][j];
      }
    return difference;
  }

  double determinant() {
    if (Double.isNaN(determinant)) {
      if (luDecompositionError) {
        determinant = 0;
      } else {
        try {
          determinant = computeDeterminant();
        } catch (final IllegalArgumentException x) { // Non-invertible
          determinant = 0;
        }
      }
    }
    return determinant;
  }

  double norm() {
    if (norm == null)
      norm = computeNorm(matrix);
    return norm;
  }

  double cond() {
    if (errorCode == ErrorCodes.NON_INVERTIBLE)
      return Double.POSITIVE_INFINITY;
    final double matrixNorm = computeNorm(matrix);
    try {
      final double inversionNorm = computeNorm(inverse());
      return matrixNorm * inversionNorm;
    } catch (final IllegalArgumentException x) { // Matrix is non-invertible
      return Double.POSITIVE_INFINITY;
    }
  }

  /* ************************************************************************************
  /*** Private methods ******************************************************************
  /**************************************************************************************/

  private static final double MIN_CORRECTION_FACTOR = 1.0 / 8;
  private static final int MAX_REFINEMENT_ITERATIONS = 20;

  /**
   * @param vector
   * @return a newly-created vector containing the solution
   */
  private double[] solveLUInternally(double[] vector) {
    if (matrix == null || vector == null || size != vector.length)
      throw new IllegalArgumentException("Matrix and vector must be of equal size");

    if (luDecompositionError)
      throwNonInvertibleError();

    if (size > 10) {
      clientDelegate.sayStarting(LU_VECTOR_SOLUTION_NAME);
      initProgressTracking(LU_VECTOR_SOLUTION_OPS_PER_M, size);
    }
    long opCount = 0;

    errorCode = ErrorCodes.OK;                                    // May remain after failed SPD decomposition
    vector = vector.clone();                                      // preserve the data passed-in

    if (luDecomposition == null) {                                // The very first invocation.
      vector = scaleMatrixAndVector(vector);          // creates luDecomposition and a scaled copy of the matrix
      opCount = decomposeLU() + scaleOpCount;                                              // modifies luDecomposition and creates the pivot.
    } else {
      vector = scaleVector(vector);                               // scales and returns vector
    }
    final double[] solution = getPermutted(vector, pivot);        // returns a new array

    // Solve L * Y = B(piv,:)
    for (int k = 0; k < size; k++) {
      for (int i = k + 1; i < size; i++) {
        solution[i] -= solution[k] * luDecomposition[i][k];        // (N^2 - N) / 2
      }
    }
    opCount += size * (size - 1) / 2;
    if (opCount > nextOpsThreshold) {
      updateProgress(opCount, size);
    }

    // Solve U * X = Y;
    for (int k = size - 1; k >= 0; k--) {
      solution[k] /= luDecomposition[k][k];                       // N
      for (int i = 0; i < k; i++) {
        solution[i] -= solution[k] * luDecomposition[i][k];       // (N^2 - N) / 2
      }
      opCount += (size - k - 1) * size;
      if (opCount > nextOpsThreshold) {
        updateProgress(opCount, size);
      }

    }

    clientDelegate.sayDone();
    return solution;
  }

  private void throwNonInvertibleError() {
    errorCode = ErrorCodes.NON_INVERTIBLE;
    this.luDecompositionError = true;
    throw new IllegalArgumentException(String.format("Matrix is unsolvable (%s)",
                                                     errorCode.name()));
  }

  /**
   * Creates luDecomposition and fills it with the scaled matrix
   * Scales and returns vector
   * @param vector
   * @return
   */
  private double[] scaleMatrixAndVector(double[] vector) {
    scaleOpCount = scaleMatrix();
    if (needToScale)
      vector = scaleVector(vector);
    return vector;
  }

  /**
   * Solves a system formed by the matrix and the given vector
   * using Cholesky decomposition
   * @param vector
   * @return the solution
   */
  private double[] solveCholeskyInternally(double[] vector) {
    if (matrix == null || vector == null || size != vector.length)
      throw new IllegalArgumentException("Matrix and vector must be of equal size");

    if (choleskyDecompositionError)
      throwCholeskyError(errorCode);

    long opCount = 0;
    if (size > 10) {
      clientDelegate.sayStarting(CH_VECTOR_SOLUTION_NAME);
      initProgressTracking(CH_VECTOR_SOLUTION_OPS_PER_M, size);
    }

    if (choleskyDecomposition == null)
      opCount = decomposeCholesky();

    final double[] solution = vector.clone();

    // Solve L*Y = B;
    for (int i = 0; i < size; i++) {
      for (int k = 0; k < i ; k++) {
        solution[i] -= solution[k] * choleskyDecomposition[i][k];
      }
      solution[i] /= choleskyDecomposition[i][i];
    }

    opCount += size * (size + 1) / 2;;
    if (opCount > nextOpsThreshold) {
      updateProgress(opCount, size);
    }

    // Solve L'*X = Y;
    for (int k = size - 1; k >= 0; k--) {
      for (int i = k + 1; i < size ; i++) {
        solution[k] -= solution[i] * choleskyDecomposition[i][k];
      }
      solution[k] /= choleskyDecomposition[k][k];
    }

    opCount += size * (size + 1) / 2;
    if (opCount > nextOpsThreshold) {
      updateProgress(opCount, size);
    }

    clientDelegate.sayDone();
    return solution;
  }

  /**
   * Iteratively refines the last found solution while the iteration count doesn't exceed
   * {@code iterationsToDo} and iterations continue to improve the accuracy.
   * The refined solution can be obtained via {@linkplain #getSolution()},
   * @return the number of iterations actually done
   */
  private void refine() {
    double[] bestSolution = solution;
    double[] deltaX = null;               // It will be filled by the result of the solution at the next iteration
    double bestError = Double.POSITIVE_INFINITY;
    double correctionFactor = 1.0;
    final int iterationsToDo = MAX_REFINEMENT_ITERATIONS;

    for (int i = 0; i < iterationsToDo; i++) {
      final double[] computedVector = multiply(solution);
      final double[] deltaB = subtractVectors(computedVector, vector);  // B1 - B0
      final double currentError = findError(deltaB);

      if (currentError < bestError) { // The best solution so far
        bestError = currentError;
        bestSolution = solution;
        if (i >= iterationsToDo || currentError == 0) {
          return;
        }
      } else {
        correctionFactor /= 2;
        if (correctionFactor < MIN_CORRECTION_FACTOR || i >=  iterationsToDo) {
          solution = bestSolution;
          return;
        }
      }

      deltaX = solve(deltaB);
      solution = correctSolutionBy(solution, deltaX, correctionFactor);
    }
  }

  /**
   * Subtracts elements of the subtrahend from corresponding elements of the minuend
   * @return minuend with the new values of the elements
   */
  private static double[] subtractVectors(double[] minuend, double[] subtrahrend) {
    for (int i = 0; i < minuend.length; i++)
      minuend[i] -= subtrahrend[i];
    return minuend;
  }

  private static double findError(double[] deviations) {
    double mse = 0;
    for (final double d: deviations) {
      mse += d * d;
    }
    return mse;
  }

  private double[] solve(double[] deltaB) {
    if (lastSolvedWith == SolutionMethod.LU)
      return solveLUInternally(deltaB);
    else
      return solveCholeskyInternally(deltaB);
  }

  /**
   * Subtracts elements of the {@code errors} each multiplied by {@code correctionFactor}
   * from corresponding elements of the the previous solution
   * @return the previous solution with the corrected values
   */
  private static double[] correctSolutionBy(double[] prevSolution, double[] errors, double correctionFactor) {
    final double[] result = new double[errors.length];
    for (int i = 0; i < errors.length; i++) {
      result[i] = prevSolution[i] - errors[i] * correctionFactor;
    }
    return result;
  }

  private static double[] multiplyElements(double[] a, double[] b) {
    final int length = a.length;
    final double[] result = new double[length];
    for (int i = 0; i < length; i++)
      result[i] = a[i] * b[i];
    return result;
  }

  // Bare Kahan summation
  private static double sumOfVector(double[] vector) {
    double sum = 0.0;
    double c = 0.0;
    for (int i = 0; i < vector.length; i++) {
      final double y = vector[i] - c;
      final double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    sumOpCount = 4 * vector.length;
    return sum;
  }

  private static long sumOpCount;

  private long scaleAndDecompose() {
    long opCount = scaleMatrix(); // creates luDecomposition and a scaled copy of matrix
    opCount += decomposeLU(); // modifies luDecomposition and creates pivot.
    return opCount;
  }

  double[][] solveAccurately(double[][] matrixB) {
    double[][] matrixX = solve(deepCopyOf(matrixB));

    double[][] deltaX = null;
    double[][] bestSolution = matrixX;
    double bestError = Double.POSITIVE_INFINITY;
    double correctionFactor = 1.0;
    final double min_correction_factor =  // 1.0/16;
                                    1.0/8;

    final int iterationsToDo = 20;
    for (int i = 0; i < iterationsToDo; i++) {
      final double[][] matrixB_1 = multiply(matrixX);
      final double currentError = subtractBfromA(matrixB_1, matrixB);

      if (currentError < bestError) {
        bestError = currentError;
        bestSolution = matrixX;
        if (i >= iterationsToDo || currentError == 0) {
          return bestSolution;
        }
      } else {
        correctionFactor /= 2;
        if (correctionFactor < min_correction_factor || i >=  iterationsToDo) {
          return bestSolution;
        }
      }

      deltaX = solve(matrixB_1);
      matrixX = subtractMatrices(matrixX, deltaX, correctionFactor);
    }

    return matrixX;
  }

  /**
   * Subtracts subtrahend from minuend and computes the MSE of the difference D as sqrt(sum(D[i][j]^2)/N)
   * @param minuend
   * @param subtrahend
   * @return
   */
  private static double subtractBfromA(double[][] minuend, double[][] subtrahend) {
    final int size = minuend.length;
    double mse = 0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        final double d = minuend[i][j] -= subtrahend[i][j];
        mse += d * d;
      }
    }
    return Math.sqrt(mse / (size * size));
  }

  private static double[][] subtractMatrices(double[][] minuend, double[][] subtrahend, double correctionFactor) {
    final int size = minuend.length;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        minuend[i][j] -= subtrahend[i][j] * correctionFactor;
      }
    }
    return minuend;
  }

  /* *****************************************************************************
   * Private methods used by solveLU *********************************************
   *******************************************************************************/

  private long decomposeLU() {
    pivot = new int[size];
    for (int i = 0; i < size; i++) {
      pivot [i] = i;
    }

    long opCount = 0;

    for (int i = 0; i < size; i++) {

      int cc = 0;

      int p = i;
      for (int j = i + 1; j < size; j++) {
        if (Math.abs(luDecomposition[j][i]) > Math.abs(luDecomposition[p][i]))
          p = j;
      }

      if (p != i) {                     // swap rows p and i
        final double[] t = luDecomposition[p]; luDecomposition[p] = luDecomposition[i]; luDecomposition[i] = t;
        final int tt = pivot[p]; pivot[p] = pivot[i]; pivot[i] = tt;
        detSignCorrection *= -1;        // Swapped rows, the determinant changes sign
      }

      final double[] row_i = luDecomposition[i];
      if (row_i[i] != 0) { // To compute some threshold here?
        final double row_i_i = 1 / row_i[i];

        for (int j = i + 1; j < size; j++) {
          final double[] row_j = luDecomposition[j];
          if (row_j[i] != 0) {
            final double row_j_i = row_j[i] *= row_i_i;
            for (int k = i + 1; k < size; k++) {
              row_j[k] -= row_i[k] * row_j_i;
            }
            cc += size - i - 1;
          }
        }
      } else {
        throwNonInvertibleError();
      }
      opCount += cc;
      if (opCount > nextOpsThreshold) {
        updateProgress(opCount, size);
      }
    } // for (int i = 0; i < size; i++) {
    return opCount;
  }

  private long decomposeCholesky() {
    choleskyDecomposition = new double[size][size];   // If we are here, decomosition == null (called from within if statement)
    long opCount = 0;

    for (int i = 0; i < size; i++) {
      double sum2 = 0;
      for (int j = 0; j < i; j++) {
        if (matrix[i][j] != matrix[j][i])
          throwCholeskyError(ErrorCodes.ASYMMETRIC);

        double sum = 0;
        for (int k = 0; k < j; k++) {
          sum += choleskyDecomposition[i][k] * choleskyDecomposition[j][k];
        }

        final double s = choleskyDecomposition[i][j] = (matrix[i][j] - sum) / choleskyDecomposition[j][j]; // 2
        sum2 += s * s;
      }

      opCount += i * (i + 1) / 2;
      if (opCount > nextOpsThreshold) {
        updateProgress(opCount, size);
      }

      final double dd = matrix[i][i] - sum2;
      if (dd <= 0 || Double.isInfinite(dd) || Double.isNaN(dd)) {
        throwCholeskyError(ErrorCodes.NON_SPD);
      }

      choleskyDecomposition[i][i] = Math.sqrt(dd);
    }
    return opCount;
  }

  /**
   */
  private void throwCholeskyError(ErrorCodes errorCode) {
    this.errorCode = errorCode;
    choleskyDecompositionError = true;
    throw new IllegalArgumentException(String.format("holeskyDecompose: matrix must be %s",
            errorCode == ErrorCodes.NON_SPD ? "positively defined" : "symmetric"));
  }

  /**
   * Puts the scaled matrix to the luDecomposition array.
   * Direct or indirect call to it is always followed by call to decomposeLU(),
   * so that luDecomposition is always either null or contains the LU-decomposition
   */
  private long scaleMatrix() {
    long opCount = 0;
    rowScales = new double[size];
    luDecomposition = new double[size][size];
    if (needToScale) {
      for (int i = 0; i < size; i++) {
        final double rowSum = sumOfAbsValues(matrix[i]);
        final double scale = (rowSum > 0) ?
            (1.0 / rowSum) : 1;
        rowScales[i] = scale;
        for (int j = 0; j < size; j++) {
          luDecomposition[i][j] = matrix[i][j] * scale;
        }
      }
      opCount += size * size;
    } else {
      for (int i = 0; i < size; i++) {
        rowScales[i] = 1.0;
        luDecomposition[i] = matrix[i].clone() ;
      }
    }
    return opCount;
  }

  /** Attention -- spoils the matrix, returns a new array */
  private double[][] scaleAndPermute(double[][] matrix, int[] indices) {
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
        matrix[i][j] = matrix[i][j] * rowScales[i];
    final double[][] result = new double[size][size];
    for (int i = 0; i < size; i++)
      result[i] = matrix[indices[i]];
    return result;
  }

  private double[] scaleVector(double[] vector) {
    for (int i = 0; i < size; i++)
      vector[i] *= rowScales[i];
    return vector;
  }

  private double sumOfAbsValues(double[] vector) {
    final double[] absValues = new double[size];
    for (int i = 0; i < size; i++)
      absValues[i] = Math.abs(vector[i]);
    return sumOfVector(absValues);
  }

  private static double[] getPermutted(double[] vector, int[] indices) {
    final double[] result = new double[vector.length];
    for (int i = 0; i < vector.length; i++)
      result[i] = vector[indices[i]];
    return result;
  }

  private double computeDeterminant() {
    if (luDecomposition == null) { // The very first invocation.
      scaleAndDecompose();
    }

    double result = 1.0;
    double scale = 1.0;

    if (needToScale) {
      for (int i = 0; i < size; i++) {
        result *= luDecomposition[i][i];
        scale *= rowScales[i];
      }
      result /= scale;
    } else {
      for (int i = 0; i < size; i++) {
        result *= luDecomposition[i][i];
      }
    }
    return result * detSignCorrection;
  }

  private double computeNorm(double[][] matrix) { // May be used to find the norm of the inversion
    double result = 0;
    for (int i = 0; i < size; i++) {
      double rowNorm = 0;
      for (int j = 0; j < size; j++) {
        rowNorm += Math.abs(matrix[i][j]);
      }
      result = Math.max(result, rowNorm);
    }
    return result;
  }

 /* Send messages about the progress to the ckient */

  private static long expectedTotalOps;
  private static long nextOpsThreshold;
  private static long opsPerPercentageStep;

  /**
   * Initializes parameters required for progress tracking.
   * The expected number of elementary arithmetic operations is computed
   * based on the matrix size and the operation type, and a threshold is
   * established for notifying the client about progress updates.
   *
   * @param opsPerM
   *        the number of elementary arithmetic operations required for the
   *        given matrix operation on a 100×100 matrix, divided by 1,000,000,000,
   *        and specific to the operation type
   */
  private static void initProgressTracking(double opsPerM, long size) {
    if (size > 10) {
      expectedTotalOps = (long)(opsPerM * size * size * size);
      nextOpsThreshold = (long)(expectedTotalOps * PERCENTAGE_STEP / 100.0); // Assuming
      opsPerPercentageStep = nextOpsThreshold;
    }
  }

  /**
   * Updates progress information based on the number of elementary operations
   * performed so far. If the configured threshold is exceeded, the client is
   * notified about the current progress, and a new threshold is computed for
   * the next notification.
   *
   * @param doneOps
   *        the number of elementary operations performed since the beginning
   *        of the long-running operation
   * @param size TODO
   */
  private static void updateProgress(long doneOps, int size) {
    if (size > 10 && doneOps >= nextOpsThreshold) {
      final double donePercent = 100.0 * doneOps / expectedTotalOps;
      clientDelegate.showPercent(donePercent);
      if (opsPerPercentageStep > 0) {
        nextOpsThreshold = (doneOps / opsPerPercentageStep + 1) * opsPerPercentageStep;
      }
    }
  }


}
