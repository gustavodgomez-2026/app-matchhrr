import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gradio import Blocks
import re
import unicodedata
import json
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuración para modelo de traducción (Español a Inglés) ---
translation_model_name = "Helsinki-NLP/opus-mt-es-en"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
# -------------------------------------------------------------------

# --- Configuración para modelo de embedding para búsqueda semántica ---
print('Cargando el modelo de embedding para búsqueda semántica...')
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print('Modelo de embedding cargado exitosamente.')
# -------------------------------------------------------------------

# --- Configuración para persistencia de datos ---
CONSULTANTS_FILE = 'consultants_data.json'
PROJECTS_FILE = 'projects_data.json'

def save_data(data, filename):
    """Guarda la lista/diccionario de datos en un archivo JSON."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_data(filename, default_value):
    """Carga datos de un archivo JSON. Retorna un valor por defecto si el archivo no existe o está vacío."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {filename}. Returning default value.")
            return default_value
    return default_value

# Inicializar datos globales cargándolos desde archivos JSON o con valores por defecto
all_consultants_data = load_data(CONSULTANTS_FILE, [])
all_stored_projects_state_initial = load_data(PROJECTS_FILE, {})
all_registered_project_names_initial = set(item.lower() for item in all_stored_projects_state_initial.keys())

extraction_columns = ['Tecnologías Conocidas', 'Experiencia Laboral (Sector)', 'Experiencia Laboral (Rol)', 'Nivel de Idiomas', 'Años de Experiencia', 'Certificaciones']

def remove_accents(input_str):
    if not isinstance(input_str, str):
        return input_str
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def extract_consultant_info(executive_summary: str) -> pd.DataFrame:
    """
    Extrae información relevante del resumen ejecutivo de un consultor utilizando reglas y expresiones regulares.
    """
    try:
        extracted_data = {
            "Tecnologías Conocidas": "N/A",
            "Experiencia Laboral (Sector)": "N/A",
            "Experiencia Laboral (Rol)": "N/A",
            "Nivel de Idiomas": "N/A",
            "Años de Experiencia": "N/A",
            "Certificaciones": "N/A"
        }

        summary_lower = remove_accents(executive_summary.lower())

        # --- Extracción de Tecnologías Conocidas ---
        known_technologies = [
            "python", "java", "c++", "c#", "javascript", "typescript", "go", "ruby", "php", "swift", "kotlin",
            "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "redis", "cassandra", "mariadb",
            "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy", "matplotlib", "seaborn", "xgboost", "lightgbm",
            "aws", "azure", "gcp", "databricks", "spark", "hadoop", "kafka", "snowflake", "tableau", "powerbi",
            "docker", "kubernetes", "git", "jenkins", "terraform", "ansible", "gitlab-ci", "argocd",
            "react", "angular", "vue", "html", "css", "nodejs", "spring", "django", "flask", "fastapi", "spring boot", "dotnet", "laravel"
        ]
        found_tech = []
        for tech in known_technologies:
            if re.search(r'\b' + re.escape(tech) + r'\b', summary_lower):
                found_tech.append(tech.capitalize())
        if found_tech:
            extracted_data["Tecnologías Conocidas"] = ", ".join(sorted(list(set(found_tech))))

        # --- Extracción de Experiencia Laboral (Sector) ---
        sectors = [
            "banca", "fintech", "retail", "ecommerce", "mineria", "gobierno", "industria",
            "salud", "telecomunicaciones", "educacion", "automotriz", "energia", "logistica", "healthtech"
        ]
        found_sectors = []
        for sector in sectors:
            if re.search(r'\b' + re.escape(sector) + r'\b', summary_lower):
                found_sectors.append(sector.capitalize())
        if found_sectors:
            extracted_data["Experiencia Laboral (Sector)"] = ", ".join(sorted(list(set(found_sectors))))

        # --- Extracción de Experiencia Laboral (Rol) ---
        # This is harder with pure regex, but we can try to find common roles
        roles = [
            "ml engineer", "data scientist", "developer", "data engineer", "genai engineer",
            "consultor", "analista", "gerente", "lider", "arquitecto", "devops", "ai researcher"
        ]
        found_roles = []
        for role in roles:
            if re.search(r'\b' + re.escape(role) + r'\b', summary_lower):
                found_roles.append(role.title())
        if found_roles:
            extracted_data["Experiencia Laboral (Rol)"] = ", ".join(sorted(list(set(found_roles))))

        # --- Extracción de Nivel de Idiomas ---
        known_languages_map = {
            "inglés": "Ingles", "ingles": "Ingles",
            "francés": "Frances", "frances": "Frances",
            "portugués": "Portugues", "portugues": "Portugues",
            "alemán": "Aleman", "aleman": "Aleman",
            "chino": "Chino", "japonés": "Japones", "japones": "Japones",
            "italiano": "Italiano", "ruso": "Ruso", "coreano": "Coreano",
            "español": "Español", "mandarín": "Mandarin", "mandarin": "Mandarin",
            "árabe": "Arabe", "arabe": "Arabe", "hindi": "Hindi"
        }
        found_languages_es = []
        for lang_es_key, lang_es_val in known_languages_map.items():
            if lang_es_key in summary_lower:
                found_languages_es.append(lang_es_val)
        if found_languages_es:
            extracted_data["Nivel de Idiomas"] = ", ".join(sorted(list(set(found_languages_es))))

        # --- Extracción de Años de Experiencia ---
        # Look for patterns like 'X años de experiencia', 'X años', 'más de X años'
        # The regex is adjusted to match 'anos' because 'remove_accents' converts 'años' to 'anos'.
        exp_match = re.search(r'(\d+(?:[.,]\d+)?|mas de \d+(?:[.,]\d+)?|más de \d+(?:[.,]\d+)?|\+\d+(?:[.,]\d+)?)\s+(?:anos|ano)', summary_lower)
        if exp_match:
            exp_str = exp_match.group(1)
            if 'mas de' in exp_str or 'más de' in exp_str or '+' in exp_str:
                extracted_data["Años de Experiencia"] = "más de 10 años" # Standardize to 'más de 10 años' for simplicity
            else:
                # The _parse_experience_to_int will handle extracting the integer part from decimals if present
                # For display, we might want to keep the decimal if it was provided, but for matching, integer is usually enough.
                # Let's keep it consistent with previous logic for now, _parse_experience_to_int handles it.
                extracted_data["Años de Experiencia"] = f"{exp_str} años"
        # Adjusting this check for 'ano' as well for singular/plural flexibility
        elif "menos de 1 ano" in summary_lower or "menos de un ano" in summary_lower:
            extracted_data["Años de Experiencia"] = "Menos de 1 año"

        # --- Extracción de Certificaciones ---
        certs = [
            "gcp", "aws", "azure", "databricks", "cgp", "pmp", "scrum master",
            "itil", "cissp", "cisa", "cism", "comptia", "oracle certified professional"
        ]
        found_certs = []
        for cert in certs:
            if re.search(r'\b' + re.escape(cert) + r'\b', summary_lower):
                found_certs.append(cert.upper() if cert in ["gcp", "aws", "azure", "cgp"] else cert.title())
        if found_certs:
            extracted_data["Certificaciones"] = ", ".join(sorted(list(set(found_certs))))

        df = pd.DataFrame([extracted_data])
        return df
    except Exception as e:
        raise gr.Error(f"Error al extraer la información del resumen: {e}")

def process_and_store_consultant(consultant_name: str, executive_summary: str, current_all_consultants_data: list) -> tuple[pd.DataFrame, list]:
    """
    Extrae información de un consultor, la añade a la lista global y devuelve el perfil individual y la lista actualizada.
    Realiza una validación para asegurar que el nombre del consultor sea único y obligatorio.
    """
    try:
        if not consultant_name.strip():
            raise gr.Error("El nombre del consultor es obligatorio.")

        if any(c.get('Nombre del Consultor', '').strip().lower() == consultant_name.strip().lower() for c in current_all_consultants_data):
            raise gr.Error(f"El consultor con el nombre '{consultant_name}' ya se encuentra registrado. Por favor, ingrese un nombre único.")

        extracted_df = extract_consultant_info(executive_summary)
        extracted_data_dict = extracted_df.iloc[0].to_dict()

        extracted_data_dict['Nombre del Consultor'] = consultant_name
        extracted_data_dict['Executive Summary Original'] = executive_summary # Store the original summary

        ordered_keys = ['Nombre del Consultor'] + [key for key in extracted_data_dict if key not in ['Nombre del Consultor', 'Executive Summary Original']] + ['Executive Summary Original']
        new_consultant_data_dict = {key: extracted_data_dict.get(key, '') for key in ordered_keys}

        new_consultant_df = pd.DataFrame([new_consultant_data_dict])

        all_display_columns = ['Nombre del Consultor'] + extraction_columns # This filters out 'Executive Summary Original' for display
        new_consultant_df = new_consultant_df[[col for col in all_display_columns if col in new_consultant_df.columns]]

        current_all_consultants_data.append(new_consultant_data_dict)
        save_data(current_all_consultants_data, CONSULTANTS_FILE) # Save after modification
        return new_consultant_df, current_all_consultants_data
    except gr.Error as e:
        raise e # Re-raise Gradio errors directly
    except Exception as e:
        raise gr.Error(f"Error al procesar y guardar el perfil del consultor: {e}")

def display_all_consultants(current_all_consultants_data: list) -> pd.DataFrame:
    """
    Convierte la lista global de consultores en un DataFrame para mostrar.
    """
    try:
        if not current_all_consultants_data:
            dummy_df_columns = ['Nombre del Consultor'] + extraction_columns
            return pd.DataFrame(columns=dummy_df_columns)

        # Filter out 'Executive Summary Original' from display if it exists
        df_to_display = pd.DataFrame(current_all_consultants_data)
        if 'Executive Summary Original' in df_to_display.columns:
            df_to_display = df_to_display.drop(columns=['Executive Summary Original'])

        return df_to_display
    except Exception as e:
        raise gr.Error(f"Error al mostrar la lista de consultores: {e}")

def _parse_experience_to_int(exp_str: str) -> int:
    if not isinstance(exp_str, str):
        return 0
    exp_str_lower = exp_str.lower().strip()

    # Handle "más de 10 años" special case first
    if "más de 10 años" in exp_str_lower or "mas de 10 años" in exp_str_lower or \
       "more than 10 years" in exp_str_lower or "+10 years" in exp_str_lower:
        return 11 # Represent >10 as a high number

    # Try converting the string directly if it's purely numeric
    if exp_str_lower.isdigit():
        return int(exp_str_lower)

    # If not purely numeric, try to find numbers within the string
    numbers = re.findall(r'\d+', exp_str_lower)
    if numbers:
        return int(numbers[0])

    return 0

def calculate_matching_score(consultant_profile: dict, project_requirements: dict) -> float:
    """
    Calcula un score de matching entre el perfil de un consultor y los requisitos de un proyecto.
    """
    try:
        score = 0.0
        max_score = 0.0

        def keyword_match(consultant_text, project_text, weight):
            nonlocal score, max_score
            max_score += weight
            if not consultant_text or not project_text:
                return
            if isinstance(project_text, list):
                project_keywords = {remove_accents(k).strip().lower() for k in project_text if k.strip()}
            else:
                project_keywords = {remove_accents(k).strip().lower() for k in project_text.split(',') if k.strip()}

            if not project_keywords:
                return

            if isinstance(consultant_text, list):
                consultant_keywords = {remove_accents(k).strip().lower() for k in consultant_text if k.strip()}
            else:
                consultant_keywords = {remove_accents(k).strip().lower() for k in consultant_text.split(',') if k.strip()}

            matches = len(consultant_keywords.intersection(project_keywords))
            if len(project_keywords) > 0:
                score += (matches / len(project_keywords)) * weight

        def experience_match(consultant_exp_str, project_exp_str, weight):
            nonlocal score, max_score
            max_score += weight

            consultant_exp = _parse_experience_to_int(consultant_exp_str)
            project_exp = _parse_experience_to_int(project_exp_str)

            if consultant_exp >= project_exp:
                score += weight
            elif consultant_exp > 0 and project_exp > 0:
                 score += (consultant_exp / project_exp) * weight * 0.5

        keyword_match(
            consultant_profile.get('Tecnologías Conocidas', ''),
            project_requirements.get('Tecnologías Requeridas', ''),
            3
        )
        keyword_match(
            consultant_profile.get('Experiencia Laboral (Sector)', ''),
            project_requirements.get('Sector del Proyecto', ''),
            2
        )
        keyword_match(
            consultant_profile.get('Nivel de Idiomas', ''),
            project_requirements.get('Idiomas Requeridos', ''),
            1.5
        )
        experience_match(
            consultant_profile.get('Años de Experiencia', ''),
            project_requirements.get('Años de Experiencia Requeridos', ''),
            2.5
        )
        keyword_match(
            consultant_profile.get('Certificaciones', ''),
            project_requirements.get('Certificaciones Necesarias', ''),
            1.5
        )

        if max_score == 0:
            return 0.0
        return (score / max_score) * 100.0
    except Exception as e:
        raise gr.Error(f"Error al calcular el score de matching: {e}")

def _create_comparison_table_df(consultant_profile: dict, project_requirements: dict) -> pd.DataFrame:
    """
    Genera un DataFrame comparativo entre los requisitos del proyecto y las habilidades del consultor.
    """
    try:
        comparison_data = []

        field_mapping = {
            "Rol del Proyecto": "Experiencia Laboral (Rol)",
            "Tecnologías Requeridas": "Tecnologías Conocidas",
            "Sector del Proyecto": "Experiencia Laboral (Sector)",
            "Idiomas Requeridos": "Nivel de Idiomas",
            "Años de Experiencia Requeridos": "Años de Experiencia",
            "Certificaciones Necesarias": "Certificaciones"
        }

        for project_key, consultant_key in field_mapping.items():
            project_value = project_requirements.get(project_key, "N/A")
            consultant_value = consultant_profile.get(consultant_key, "N/A")

            comparison_data.append({
                "Categoría": project_key.replace(" Requeridas", "").replace(" Requerido", "").replace(" del Proyecto", ""), # Clean up category names for display
                "Requisito del Proyecto": project_value if project_value else "N/A",
                "Habilidad del Consultor": consultant_value if consultant_value else "N/A"
            })

        return pd.DataFrame(comparison_data)
    except Exception as e:
        raise gr.Error(f"Error al crear la tabla comparativa: {e}")


def translate_summary_for_top_consultant(top_match_info):
    """
    Genera la traducción al inglés del resumen ejecutivo del consultor con el match más alto.
    """
    if not top_match_info or not top_match_info.get('top_consultant'):
        gr.Warning("Primero calcula el matching para seleccionar un candidato.")
        return ""

    original_summary = top_match_info['top_consultant']['profile'].get('Executive Summary Original', '')
    if not original_summary:
        return "No se encontró el resumen ejecutivo original para traducir."

    try:
        # Perform translation manually using the model and tokenizer
        inputs = translation_tokenizer(original_summary, return_tensors="pt")
        translated_tokens = translation_model.generate(**inputs)
        translated_text = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        gr.Error(f"Error durante la traducción: {e}")
        return "Error al traducir el resumen ejecutivo."


def semantic_search_consultants(query: str, all_consultants_data: list):
    """
    Realiza una búsqueda semántica de consultores basada en una consulta en lenguaje natural.
    """
    if not query.strip():
        gr.Warning("Por favor, ingresa una consulta para la búsqueda semántica.")
        return pd.DataFrame(columns=["Nombre del Consultor", "Similitud Semántica"])

    if not all_consultants_data:
        gr.Warning("No hay consultores registrados para realizar la búsqueda semántica.")
        return pd.DataFrame(columns=["Nombre del Consultor", "Similitud Semántica"])

    try:
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)

        consultant_results = []
        for consultant in all_consultants_data:
            # Create a combined string of skills for embedding
            skills_summary = (
                f"{consultant.get('Tecnologías Conocidas', '')} "
                f"{consultant.get('Experiencia Laboral (Sector)', '')} "
                f"{consultant.get('Experiencia Laboral (Rol)', '')} "
                f"{consultant.get('Nivel de Idiomas', '')} "
                f"{consultant.get('Años de Experiencia', '')} "
                f"{consultant.get('Certificaciones', '')}"
            )
            skills_summary = " ".join(filter(None, skills_summary.split())) # Remove extra spaces and empty strings

            if not skills_summary:
                continue # Skip consultants with no skills information

            consultant_embedding = sentence_model.encode(skills_summary, convert_to_tensor=True)

            similarity = cosine_similarity(query_embedding.unsqueeze(0), consultant_embedding.unsqueeze(0)).item()

            consultant_results.append({
                "Nombre del Consultor": consultant.get('Nombre del Consultor', 'Desconocido'),
                "Similitud Semántica": round(similarity * 100, 2) # Convert to percentage
            })

        df_results = pd.DataFrame(consultant_results).sort_values(by="Similitud Semántica", ascending=False)
        return df_results
    except Exception as e:
        raise gr.Error(f"Error durante la búsqueda semántica: {e}")


with Blocks(title="Matching de Consultores") as demo:
    all_consultants_state = gr.State(all_consultants_data)
    all_stored_projects_state = gr.State(all_stored_projects_state_initial)
    all_registered_project_names = gr.State(all_registered_project_names_initial)

    gr.Markdown("# Extractor y Gestor de Perfiles de Consultores")

    with gr.Tab("Cargar Perfil"):
        gr.Markdown("## Introduce el Resumen Ejecutivo del Consultor")
        consultant_name_input = gr.Textbox(
            label="Nombre del Consultor (Obligatorio)",
            placeholder="Ingresa el nombre completo del consultor...",
            interactive=True
        )
        summary_input = gr.Textbox(lines=10, placeholder="Pega aquí el resumen ejecutivo del consultor...")
        extract_button = gr.Button("Extraer y Guardar Perfil")

        individual_df_columns = ['Nombre del Consultor'] + extraction_columns
        output_df_individual = gr.Dataframe(headers=individual_df_columns, label="Perfil del Consultor Extraído")

    with gr.Tab("Consultores Disponibles"):
        gr.Markdown("## Todos los Consultores Registrados")
        refresh_button = gr.Button("Actualizar Lista de Consultores")

        all_df_columns = ['Nombre del Consultor'] + extraction_columns
        output_df_all = gr.Dataframe(headers=all_df_columns, label="Base de Datos de Consultores")

    with gr.Tab("Cargar Proyecto"):
        gr.Markdown("## Ingresa los Requisitos del Proyecto")
        project_name_input = gr.Textbox(
            label="Nombre del Proyecto (Obligatorio)",
            placeholder="Ingresa el nombre del proyecto...",
            interactive=True
        )
        project_role_input = gr.Dropdown(
            label="Rol del Proyecto",
            choices=["Data Scientist", "ML Ops", "Developer", "Data Engineer", "GenAI Engineer"],
            multiselect=False
        )
        project_tech_input = gr.Dropdown(
            label="Tecnologías Requeridas",
            choices=["Python", "TensorFlow", "SQL", "Javascript", "React", "HTML"],
            multiselect=True
        )
        project_sector_input = gr.Dropdown(
            label="Sector del Proyecto",
            choices=["Banca/Fintech", "Retail", "Ecommerce", "Mineria", "Gobierno", "Industria"],
            multiselect=False
        )
        project_languages_input = gr.Dropdown(
            label="Idiomas Requeridos",
            choices=["Ingles", "Frances", "Portugues"],
            multiselect=True
        )
        project_experience_input = gr.Dropdown(
            label="Años de Experiencia Requeridos",
            choices=[f"{i} año" if i == 1 else f"{i} años" for i in range(1, 11)] + ["más de 10 años"],
            multiselect=False
        )
        project_certs_input = gr.Dropdown(
            label="Certificaciones Necesarias",
            choices=["AWS", "GCP", "Azure", "Databricks"],
            multiselect=True
        )
        process_project_button = gr.Button("Guardar Requisitos del Proyecto")
        output_project_requirements = gr.Dataframe(headers=[
            "Nombre del Proyecto", "Rol del Proyecto", "Tecnologías Requeridas", "Sector del Proyecto",
            "Idiomas Requeridos", "Años de Experiencia Requeridos",
            "Certificaciones Necesarias"
        ], label="Requisitos del Proyecto Guardados")

    with gr.Tab("Matching de Consultores"):
        gr.Markdown("## Realizar Matching de Consultores con el Proyecto Actual")
        project_selection_dropdown = gr.Dropdown(
            label="Selecciona un Proyecto",
            choices=[],
            interactive=True
        )
        match_button = gr.Button("Calcular Scores de Matching")
        output_matching_results = gr.Dataframe(headers=["Nombre del Consultor", "Score de Matching"], label="Resultado del Matching")

        justification_button = gr.Button("Comparativa detallada consultor y proyecto")
        # This Markdown component will now serve as the dynamic HTML title for the comparison table
        output_dynamic_comparison_title = gr.Markdown(value="## Comparativa Proyecto vs. Consultor", visible=True)
        output_comparison_table = gr.Dataframe(visible=True, wrap=True, datatype="html") # No 'label' here; it's handled by output_dynamic_comparison_title

        gr.Markdown("## Accede al resumen ejecutivo del consultor seleccionado en INGLES")
        translate_summary_button = gr.Button("Traducir Resumen Ejecutivo al Inglés")
        output_translated_summary = gr.Textbox(label="Resumen Ejecutivo Traducido (Inglés)", lines=5, interactive=False)

        gr.Markdown("## Búsqueda Otros Posibles Candidatos")
        semantic_search_query_input = gr.Textbox(
            label="Ingresa lo que buscas en lenguaje natural",
            placeholder="Ej: Busco un científico de datos con experiencia en machine learning y sector bancario...",
            lines=3
        )
        semantic_search_button = gr.Button("Buscar Candidatos por Similitud Semántica")
        output_semantic_search_results = gr.Dataframe(
            headers=["Nombre del Consultor", "Similitud Semántica"],
            label="Resultados de Búsqueda Semántica"
        )

    extract_button.click(
        fn=process_and_store_consultant,
        inputs=[consultant_name_input, summary_input, all_consultants_state],
        outputs=[output_df_individual, all_consultants_state]
    ).success(
        fn=lambda x: x,
        inputs=all_consultants_state,
        outputs=output_df_all
    )

    refresh_button.click(
        fn=display_all_consultants,
        inputs=all_consultants_state,
        outputs=output_df_all
    )

    def store_project_requirements(project_name, role, tech, sector, languages, experience, certifications, current_registered_names_set, current_all_stored_projects_dict):
        try:
            if not project_name.strip():
                raise gr.Error("El nombre del proyecto es obligatorio.")
            normalized_project_name = project_name.strip().lower()
            if normalized_project_name in current_registered_names_set:
                raise gr.Error(f"Ya existe un proyecto con el nombre '{project_name}'. Por favor, ingrese un nombre único.")

            tech_str = ", ".join(tech) if isinstance(tech, list) else tech
            languages_str = ", ".join(languages) if isinstance(languages, list) else languages
            certifications_str = ", ".join(certifications) if isinstance(certifications, list) else certifications

            project_data = {
                "Nombre del Proyecto": project_name,
                "Rol del Proyecto": role,
                "Tecnologías Requeridas": tech_str,
                "Sector del Proyecto": sector,
                "Idiomas Requeridos": languages_str,
                "Años de Experiencia Requeridos": experience,
                "Certificaciones Necesarias": certifications_str
            }

            updated_registered_names_set = current_registered_names_set.copy()
            updated_registered_names_set.add(normalized_project_name)

            updated_all_stored_projects_dict = current_all_stored_projects_dict.copy()
            updated_all_stored_projects_dict[project_name] = project_data

            save_data(updated_all_stored_projects_dict, PROJECTS_FILE) # Save after modification
            return pd.DataFrame([project_data]), updated_all_stored_projects_dict, updated_registered_names_set
        except gr.Error as e:
            raise e # Re-raise Gradio errors directly
        except Exception as e:
            raise gr.Error(f"Error al guardar los requisitos del proyecto: {e}")

    def update_project_selection_dropdown(all_projects_dict):
        return gr.update(choices=list(all_projects_dict.keys()))

    process_project_button.click(
        fn=store_project_requirements,
        inputs=[
            project_name_input,
            project_role_input, project_tech_input, project_sector_input,
            project_languages_input, project_experience_input,
            project_certs_input,
            all_registered_project_names,
            all_stored_projects_state
        ],
        outputs=[output_project_requirements, all_stored_projects_state, all_registered_project_names]
    ).success(
        fn=update_project_selection_dropdown,
        inputs=all_stored_projects_state,
        outputs=project_selection_dropdown
    )

    current_top_match_info_state = gr.State(None)

    def perform_matching(all_consultants_data_list, all_stored_projects_dict, selected_project_name):
        try:
            if not all_consultants_data_list:
                gr.Warning("No hay consultores registrados. Por favor, cargue perfiles primero.")
                # Return empty outputs for both output_matching_results and current_top_match_info_state
                # and ensure output_dynamic_comparison_title is reset
                return pd.DataFrame(columns=['Nombre del Consultor', 'Score de Matching']), None, "## Comparativa Proyecto vs. Consultor"
            if not selected_project_name:
                gr.Warning("Por favor, seleccione un proyecto para el matching.")
                return pd.DataFrame(columns=['Nombre del Consultor', 'Score de Matching']), None, "## Comparativa Proyecto vs. Consultor"

            project_reqs_dict = all_stored_projects_dict.get(selected_project_name)
            if not project_reqs_dict:
                gr.Warning(f"No se encontraron requisitos para el proyecto '{selected_project_name}'.")
                return pd.DataFrame(columns=['Nombre del Consultor', 'Score de Matching']), None, "## Comparativa Proyecto vs. Consultor"

            results = []
            for consultant_dict in all_consultants_data_list:
                score = calculate_matching_score(consultant_dict, project_reqs_dict)
                results.append({
                    'Nombre del Consultor': consultant_dict.get('Nombre del Consultor', 'Desconocido'),
                    'Score de Matching': round(score, 2),
                    'full_profile': consultant_dict
                })

            df_results = pd.DataFrame(results).sort_values(by='Score de Matching', ascending=False)

            top_consultant_for_pitch = None
            max_score = 0.0

            if not df_results.empty:
                max_score = df_results['Score de Matching'].max()
                top_consultants = df_results[df_results['Score de Matching'] == max_score]

                if not top_consultants.empty:
                    first_top_consultant_row = top_consultants.iloc[0]
                    top_consultant_for_pitch = {
                        'name': first_top_consultant_row['Nombre del Consultor'],
                        'profile': first_top_consultant_row['full_profile'],
                        'score': first_top_consultant_row['Score de Matching']
                    }

                def highlight_max_score_cells(row):
                    styles = [''] * len(row)
                    name_idx = row.index.get_loc('Nombre del Consultor')
                    score_idx = row.index.get_loc('Score de Matching')

                    if row['Score de Matching'] == max_score:
                        styles[name_idx] = 'color: orange; font-weight: bold'
                        styles[score_idx] = 'color: orange; font-weight: bold'
                    return styles

                styled_df = df_results.drop(columns=['full_profile']).style.apply(highlight_max_score_cells, axis=1)
                return styled_df, {'top_consultant': top_consultant_for_pitch, 'project_reqs': project_reqs_dict, 'selected_project_name': selected_project_name}, "## Comparativa Proyecto vs. Consultor"

            return pd.DataFrame(columns=['Nombre del Consultor', 'Score de Matching']), None, "## Comparativa Proyecto vs. Consultor"
        except Exception as e:
            raise gr.Error(f"Error al realizar el matching: {e}")

    def generate_sales_pitch(top_match_info):
        try:
            print("--- Inside generate_sales_pitch ---")

            if not top_match_info or not top_match_info.get('top_consultant'):
                gr.Warning("Primero calcula el matching para seleccionar un candidato.")
                print("No top consultant info. Returning empty string.")
                # Reset the dynamic title as well when no top consultant is found
                return "## Comparativa Proyecto vs. Consultor", pd.DataFrame()

            profile = top_match_info['top_consultant']['profile']
            project_reqs_dict = top_match_info['project_reqs']
            top_consultant_name = top_match_info['top_consultant']['name']
            selected_project_name = top_match_info['selected_project_name']

            # Create the base comparison DataFrame
            comparison_df = _create_comparison_table_df(profile, project_reqs_dict)

            # Apply granular highlighting by modifying cell content to HTML
            highlighted_df = comparison_df.copy()
            highlight_style = "background-color: #FFF3E0; font-weight: bold; color: #E65100;"

            # Helper to format text with highlighting
            def format_with_highlight(text_to_format, keywords_to_highlight_normalized):
                if not text_to_format or text_to_format == "N/A":
                    return text_to_format

                # Split by comma and clean parts. Preserve original casing for display.
                parts = [p.strip() for p in text_to_format.split(',') if p.strip()]
                formatted_parts = []
                for part in parts:
                    if remove_accents(part).lower() in keywords_to_highlight_normalized:
                        formatted_parts.append(f"<span style='{highlight_style}'>{part}</span>")
                    else:
                        formatted_parts.append(part)
                return ", ".join(formatted_parts)

            # Helper to normalize experience strings for robust comparison
            def normalize_experience_string_for_match(s):
                if not isinstance(s, str):
                    return ""
                s = remove_accents(s) # Use the existing global remove_accents
                s = s.replace(' - ', ' ') # Handle potential spaces around hyphens
                s = s.lower().strip() # Convert to lowercase and strip whitespace
                return s

            for index, row in comparison_df.iterrows():
                category = row['Categoría']
                project_req_display_val = row['Requisito del Proyecto'] # Value as displayed in table
                consultant_skill_display_val = row['Habilidad del Consultor'] # Value as displayed in table

                # Retrieve original, unformatted values from dictionaries/profile for robust keyword extraction
                original_project_req_val_from_dict = None
                consultant_original_skill_val_from_profile = None

                if category == 'Rol':
                    original_project_req_val_from_dict = project_reqs_dict.get('Rol del Proyecto', '')
                    consultant_original_skill_val_from_profile = profile.get('Experiencia Laboral (Rol)', '')
                elif category == 'Tecnologías':
                    original_project_req_val_from_dict = project_reqs_dict.get('Tecnologías Requeridas', '')
                    consultant_original_skill_val_from_profile = profile.get('Tecnologías Conocidas', '')
                elif category == 'Sector':
                    original_project_req_val_from_dict = project_reqs_dict.get('Sector del Proyecto', '')
                    consultant_original_skill_val_from_profile = profile.get('Experiencia Laboral (Sector)', '')
                elif category == 'Idiomas':
                    original_project_req_val_from_dict = project_reqs_dict.get('Idiomas Requeridos', '')
                    consultant_original_skill_val_from_profile = profile.get('Nivel de Idiomas', '')
                elif category == 'Años de Experiencia':
                    original_project_req_val_from_dict = project_reqs_dict.get('Años de Experiencia Requeridos', '')
                    consultant_original_skill_val_from_profile = profile.get('Años de Experiencia', '')
                elif category == 'Certificaciones':
                    original_project_req_val_from_dict = project_reqs_dict.get('Certificaciones Necesarias', '')
                    consultant_original_skill_val_from_profile = profile.get('Certificaciones', '')

                if category in ['Tecnologías', 'Idiomas', 'Certificaciones', 'Sector']:
                    project_keywords_set_normalized = {remove_accents(k).strip().lower() for k in str(original_project_req_val_from_dict).split(',') if k.strip()}
                    consultant_keywords_set_normalized = {remove_accents(k).strip().lower() for k in str(consultant_original_skill_val_from_profile).split(',') if k.strip()}

                    if category == 'Idiomas':
                        print(f"DEBUG: Idiomas - Project Req (raw): '{original_project_req_val_from_dict}'")
                        print(f"DEBUG: Idiomas - Consultant Skill (raw): '{consultant_original_skill_val_from_profile}'")
                        print(f"DEBUG: Idiomas - Project Keywords Normalized: {project_keywords_set_normalized}")
                        print(f"DEBUG: Idiomas - Consultant Keywords Normalized: {consultant_keywords_set_normalized}")

                    # Find common keywords for highlighting in both columns
                    overlapping_keywords = project_keywords_set_normalized.intersection(consultant_keywords_set_normalized)
                    if category == 'Idiomas':
                        print(f"DEBUG: Idiomas - Overlapping Keywords: {overlapping_keywords}")

                    highlighted_df.loc[index, 'Requisito del Proyecto'] = format_with_highlight(project_req_display_val, overlapping_keywords)
                    highlighted_df.loc[index, 'Habilidad del Consultor'] = format_with_highlight(consultant_skill_display_val, overlapping_keywords)

                elif category == 'Rol':
                    project_role_norm = remove_accents(str(original_project_req_val_from_dict)).strip().lower()
                    consultant_role_norm = remove_accents(str(consultant_original_skill_val_from_profile)).strip().lower()

                    if project_role_norm and consultant_role_norm and project_role_norm in consultant_role_norm:
                        highlighted_df.loc[index, 'Requisito del Proyecto'] = f"<span style='{highlight_style}'>{project_req_display_val}</span>"
                        # For simplicity, if there's an exact match for role, highlight the whole cell
                        if project_role_norm == consultant_role_norm:
                            highlighted_df.loc[index, 'Habilidad del Consultor'] = f"<span style='{highlight_style}'>{consultant_skill_display_val}</span>"
                        else:
                            highlighted_df.loc[index, 'Habilidad del Consultor'] = consultant_skill_display_val

                elif category == 'Años de Experiencia':
                    print(f"Debug 'Años de Experiencia' - Project Req Display: '{project_req_display_val}'")
                    print(f"Debug 'Años de Experiencia' - Consultant Skill Display: '{consultant_skill_display_val}'")

                    normalized_consultant_exp_for_comparison = _parse_experience_to_int(consultant_skill_display_val)
                    normalized_project_exp_for_comparison = _parse_experience_to_int(project_req_display_val)

                    print(f"Debug 'Años de Experiencia' - Normalized Project: '{normalized_project_exp_for_comparison}'")
                    print(f"Debug 'Años de Experiencia' - Normalized Consultant: '{normalized_consultant_exp_for_comparison}'")

                    if normalized_project_exp_for_comparison > 0 and normalized_consultant_exp_for_comparison >= normalized_project_exp_for_comparison:
                        highlighted_df.loc[index, 'Requisito del Proyecto'] = f"<span style='{highlight_style}'>{project_req_display_val}</span>"
                        highlighted_df.loc[index, 'Habilidad del Consultor'] = f"<span style='{highlight_style}'>{consultant_skill_display_val}</span>"

            dynamic_label_html = f"<span style='font-size: 1.2em; font-weight: bold; color: orange;'>Match entre {selected_project_name} & {top_consultant_name}</span>"
            return dynamic_label_html, highlighted_df
        except Exception as e:
            raise gr.Error(f"Error al generar el argumento de venta: {e}")

    match_button.click(
        fn=perform_matching,
        inputs=[all_consultants_state, all_stored_projects_state, project_selection_dropdown],
        outputs=[output_matching_results, current_top_match_info_state, output_dynamic_comparison_title] # Changed output here
    )

    justification_button.click(
        fn=generate_sales_pitch,
        inputs=[current_top_match_info_state],
        outputs=[output_dynamic_comparison_title, output_comparison_table] # Changed output here
    )

    translate_summary_button.click(
        fn=translate_summary_for_top_consultant,
        inputs=[current_top_match_info_state],
        outputs=[output_translated_summary]
    )

    semantic_search_button.click(
        fn=semantic_search_consultants,
        inputs=[semantic_search_query_input, all_consultants_state],
        outputs=[output_semantic_search_results]
    )

    project_selection_dropdown.change(
        fn=lambda project_name: gr.update(value=f"Identificar el consultor con mejor match para {project_name}") if project_name else gr.update(value="Calcular Scores de Matching"),
        inputs=[project_selection_dropdown],
        outputs=[match_button]
    )

    # Modify the load events to use the persisted data
    demo.load(display_all_consultants, inputs=all_consultants_state, outputs=output_df_all)
    demo.load(update_project_selection_dropdown, inputs=all_stored_projects_state, outputs=project_selection_dropdown)

demo.launch(debug=True)
