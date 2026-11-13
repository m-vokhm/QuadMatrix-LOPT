# CLAUDE.md

## Проект

QuadMatrix-LOPT — Java-библиотека для вычислений с матрицами.

- `Matrix` — общий абстрактный предок.
- `DoubleMatrix` — вычисления на `double`, использует `DoubleMatrixSolver`.
- `QuadrupleMatrix` — вычисления на `Quadruple` (четверная точность), использует `QuadrupleMatrixSolver`.
- `BigDecimalMatrix` / `BigDecimalMatrixSolver` — пока не используются, не трогать.

Решатели наследуются от `MatrixSolver` (пакет `com.mvohm.quadmatrix`).

## Текущая задача

Оповещение клиента о ходе вычислений через делегат `api.ClientDelegate`:
`sayStarting(opName)` в начале операции, `sayDone()` в конце,
`showPercent(percent)` для прогресса.

Эталонная реализация — `DoubleMatrixSolver`:
- `sayStarting()` / `sayDone()` в `solve(...)`, `inverse()`, `multiply(matrix)`,
  `solveLUInternally()`, `solveCholeskyInternally()`;
- `initProgressTracking(opsPerM, size)` и `updateProgress(doneOps, size)` в
  `doSolve()`, `multiply(matrix)`, `solveLUInternally()`, `solveCholeskyInternally()`,
  а `updateProgress()` также в `decomposeLU()` и `decomposeCholesky()`;
- в этих методах ведётся счётчик `opCount`; при превышении `nextOpsThreshold`
  вызывается `updateProgress()`, который сдвигает порог.

Задача — перенести этот механизм в `QuadrupleMatrixSolver` по аналогии.

## Правила работы

- Сборку, тесты и коммиты выполняет пользователь; Claude их не делает, если не попросят.
- Не читать весь проект без необходимости — только нужные файлы.
- Отчёты о работе — в каталоге `Claude/`, файлы вида `WN_YYYY_MM_DD.md`.
