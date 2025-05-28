## 📌 Preguntas y Respuestas del Proyecto

### 1. ¿Qué criterio utilizaste para seleccionar la variable independiente en la regresión lineal simple? ¿Por qué crees que esa variable influye en el monto del préstamo?

Se usó el **sueldo promedio** del cliente como variable independiente porque está directamente relacionado con su capacidad de pago. En general, una persona con mayor ingreso puede acceder a préstamos más altos, lo cual se reflejaba en los datos del dataset.

---

### 2. ¿Qué significa el error MSE (Mean Squared Error) y qué te indica sobre el rendimiento del modelo?

El **MSE (Mean Squared Error)** mide el promedio del cuadrado de los errores entre valores predichos y reales. Un valor bajo indica buen ajuste del modelo, mientras que uno alto sugiere que las predicciones no están muy cerca de los valores verdaderos. En este caso, el MSE mostró que el modelo tiene margen de mejora.

---

### 3. ¿Qué conclusiones sacaste al comparar los valores predichos por el modelo con los valores reales? ¿Hubo sobreajuste o subajuste? ¿Cómo lo detectaste?

Al graficar los valores reales contra los predichos, se observó que el modelo no sigue bien la dispersión de los datos. Esto indicó un **subajuste**, ya que el modelo es demasiado simple para capturar la relación entre las variables. No hubo signos claros de sobreajuste.

---

### 4. ¿Qué desafíos encontraste al trabajar con el dataset y cómo los solucionaste?

Entre los principales desafíos estuvieron:
- **Valores faltantes:** Se eliminaron filas inconsistentes.
- **Formato incorrecto en columnas numéricas:** Se corrigieron comas y se convirtieron a tipo `float`.
- **Variables categóricas:** Se codificaron usando `OneHotEncoder`.
- **Rangos de sueldo como texto:** Se transformaron en números con una función personalizada.
- **Inconsistencia entre X e Y:** Se limpiaron juntos para garantizar consistencia.

Todos estos problemas se resolvieron mediante técnicas de preprocesamiento con `pandas`, `scikit-learn` y funciones personalizadas.