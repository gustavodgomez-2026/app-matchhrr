# app-matchhrr
**App para identificar los mejores consultores para un proyecto**

Esta aplicación interactiva de Gradio está diseñada para facilitar la gestión y el matching de perfiles de consultores con los requisitos específicos de proyectos. En la aplicación se utilizan modelos de **Traducción** y de **Búsqueda Semántica**, además de otras funcionalidades como **extracción basada en reglas** y **scoring inteligente** para calcular el match entre proyecto y consultor. 

## Descripción del Proyecto, Tecnologías y Ejecución

### 1. Descripción del Proyecto: Matching de Consultores

Esta aplicación web interactiva, construida con Gradio, está diseñada para facilitar el proceso de *matching* entre consultores y proyectos. Su objetivo principal es automatizar la extracción de información clave de los resúmenes ejecutivos de los consultores y permitir a los usuarios definir los requisitos de los proyectos. Con base en esta información estructurada, la aplicación calcula un score de compatibilidad, presentando los resultados de manera clara y justificando el *matching* con un detalle comparativo visual.

**¿Qué hace?**
*   **Extracción de Perfiles de Consultores:** A partir de un resumen ejecutivo en texto libre, la aplicación extrae automáticamente datos como tecnologías conocidas, experiencia laboral (sector y rol), nivel de idiomas, años de experiencia y certificaciones.
*   **Gestión de Proyectos:** Permite definir y almacenar los requisitos específicos de los proyectos (rol, tecnologías, sector, idiomas, experiencia, certificaciones).
*   **Cálculo de Matching:** Evalúa la compatibilidad de cada consultor con un proyecto seleccionado, asignando un score porcentual.
*   **Justificación Detallada:** Genera una tabla comparativa visualmente enriquecida que resalta las coincidencias entre las habilidades del consultor y los requisitos del proyecto, facilitando la toma de decisiones.
*   **Búsqueda Semántica:** Permite buscar consultores utilizando consultas en lenguaje natural, encontrando a los más relevantes basándose en la similitud semántica de sus perfiles.
*   **Traducción de Resúmenes:** Ofrece la funcionalidad de traducir el resumen ejecutivo del consultor con el mejor *matching* a otros idiomas (inglés, en este caso).

**¿Por qué es útil?**
Esta herramienta optimiza el tiempo en la selección de personal para proyectos, reduce el sesgo en la evaluación de candidatos y proporciona una justificación objetiva para las decisiones de *staffing*. Es ideal para consultoras, empresas de reclutamiento o departamentos de gestión de proyectos que buscan una forma eficiente y estructurada de gestionar sus recursos humanos.

### 2. Tecnologías Utilizadas

La aplicación se basa en una combinación de librerías de Python para procesamiento de lenguaje natural, interfaz de usuario y manejo de datos:

*   **Gradio:** Framework principal para la construcción de la interfaz de usuario web interactiva. Permite crear rápidamente demos y aplicaciones web para modelos de Machine Learning y funciones arbitrarias de Python.
*   **Hugging Face Transformers:** Se utiliza para la traducción de texto (modelo `Helsinki-NLP/opus-mt-es-en`) y, en el pasado, fue considerado para la extracción de información (aunque actualmente se utilizan más expresiones regulares y reglas para la extracción estructurada).
*   **Sentence-Transformers:** Esencial para la funcionalidad de búsqueda semántica. Permite generar embeddings vectoriales de los textos (resúmenes de consultores y consultas de búsqueda) para luego calcular la similitud coseno entre ellos (modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
*   **Pandas:** Utilizado extensivamente para el manejo y manipulación de datos, especialmente para la creación y visualización de DataFrames que representan perfiles de consultores, requisitos de proyectos y resultados de *matching*.
*   **Scikit-learn (sklearn.metrics.pairwise.cosine_similarity):** Empleado para calcular la similitud coseno entre los embeddings generados por `sentence-transformers`.
*   **JSON:** Utilizado para la persistencia de datos (perfiles de consultores y proyectos) en archivos locales (`consultants_data.json`, `projects_data.json`), asegurando que la información se mantenga entre sesiones.
*   **Python `re` (Regular Expressions):** Fundamental para la extracción de información estructurada de los resúmenes ejecutivos, permitiendo identificar patrones específicos (ej. años de experiencia, tecnologías).

### 3. Instrucciones de Instalación y Ejecución Local

Para instalar y ejecutar la aplicación de Gradio en tu entorno local, sigue los siguientes pasos:

1.  **Clonar el Repositorio (si aplica):**
    Si el código está en un repositorio Git, clónalo a tu máquina local:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2.  **Crear un Entorno Virtual (Recomendado):**
    Es una buena práctica utilizar un entorno virtual para gestionar las dependencias del proyecto y evitar conflictos con otras instalaciones de Python.
    ```bash
    python -m venv venv
    ```

3.  **Activar el Entorno Virtual:**
    *   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Instalar las Dependencias:**
    Con el entorno virtual activado, instala las librerías necesarias. Puedes usar el archivo `requirements.txt` si está disponible, o instalarlas manualmente:
    
    **Opción A: Usando `requirements.txt`**
    ```bash
    pip install -r requirements.txt
    ```
    
    **Opción B: Instalación Manual**
    ```bash
    pip install gradio transformers pandas==2.2.2 sentence-transformers torch
    ```
    
    *Nota: `pandas==2.2.2` se especifica para asegurar compatibilidad. `torch` es una dependencia de `sentence-transformers`.*

5.  **Configurar el Token de Hugging Face (Opcional pero Recomendado):**
    Si vas a interactuar con modelos de Hugging Face que requieren autenticación, es buena idea configurar tu `HF_TOKEN`. Puedes establecerlo como una variable de entorno:
    ```bash
    export HF_TOKEN="tu_token_de_huggingface"
    # Para Windows (PowerShell):
    # $env:HF_TOKEN="tu_token_de_huggingface"
    ```
    En Colab, esto se maneja a través de los Secrets. Localmente, una variable de entorno es la forma más común.

6.  **Ejecutar la Aplicación Gradio:**
    Navega al directorio donde se encuentra tu script principal (por ejemplo, `app.py` o el notebook convertido a `.py`). Luego, ejecuta el script de Python:
    ```bash
    python tu_script_principal.py
    ```
    Si estás ejecutando directamente el notebook en un entorno Jupyter (como JupyterLab o un entorno de desarrollo local), simplemente abre el notebook y ejecuta todas las celdas.

    Una vez ejecutado, Gradio imprimirá una URL en tu terminal (usualmente `http://127.0.0.1:7860/` o similar). Abre esta URL en tu navegador web para acceder a la aplicación.
