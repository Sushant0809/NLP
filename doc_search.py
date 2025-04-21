import streamlit as st
import json
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain.llms import Ollama
import os

# Initialize the SentenceTransformer model and LLM
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device='cuda', trust_remote_code=True)
# model = SentenceTransformer("/project_resources/shared_docs/gte-Qwen2-1.5B-instruct", device='cuda', trust_remote_code=True)
llm = Ollama(model="phi4", temperature=0.1)

# Cache the loading of patient note data for faster performance
@st.cache_data
def load_res(directory):
    res = {}
    filepaths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    for file in filepaths:
        pid = os.path.basename(file).split('.')[0]
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data['structured'], dict):
            res[pid] = data['structured']
    return res

# Directory path to the structured results
directory = "/project_resources/common/retriever_v2/PMC_EXPT2/structured_results_pmc_dataset"  # Replace with your directory path
res = load_res(directory)

def search_embeddings(query, collection_name, top_k=10):
    """
    Search for similar embeddings in a Milvus collection.
    """
    query_embedding = model.encode(query, convert_to_tensor=False, show_progress_bar=False)
    milvus_host = "localhost"
    milvus_port = "19530"
    connections.connect("default", host=milvus_host, port=milvus_port)
    collection = Collection(collection_name)
    collection.load()
    search_params = {"metric_type": "COSINE"}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["patient_id", "patient_text"]
    )
    chk = []
    chk_txt = []
    for result in results[0]:
        chk.append((result.entity.get('patient_id'), result.distance))
        chk_txt.append({result.entity.get('patient_id'): result.entity.get('patient_text')})
    return chk, chk_txt

def construct_prompt_category(patient_note: str) -> str:
    chat_prompt = f"""<|im_start|>system<|im_sep|>
You are an expert in patient note categorization. Analyze the patient note provided below and determine which of the following categories it belongs to. For each category that applies, assign a confidence score representing how strongly the patient note matches that category, and include the part of the patient note that corresponds to the category. Only use the categories listed below:

- Demographic Information: Details about the patient’s age, sex, and other demographic characteristics.
- Clinical History and Presentation: Information on the patient’s symptoms, chief complaints, onset, and progression.
- Anatomical Location: Specific details identifying the site of the lesion or abnormality.
- Imaging and Radiologic Findings: Findings from imaging studies (e.g., CT, radiographs) including key features.
- Histopathological and Diagnostic Findings: Descriptions of biopsy results, histopathology, and other diagnostic details.
- Molecular and Genetic Data: Information regarding molecular markers, immunohistochemistry, genetic tests, or related data.
- Treatment Modalities: Details about surgical interventions, chemotherapy, radiotherapy, or other treatments.
- Clinical Outcome and Follow-Up: Information regarding treatment response, progression, recurrence, and outcome.
- Unassigned: For sentences that do not clearly fit into any of the above categories.

For each category that applies, return a valid Python dictionary with the following structure:
```python
{{
    "Category1": {{"score": "score1", "text": "relevant part of the patient note for Category1"}},
    "Category2": {{"score": "score2", "text": "relevant part of the patient note for Category2"}},
    ...
}}
The confidence score should be a float between 0 and 1 (with two decimal places precision), and the text should be the part of the patient note that corresponds to the category. Provide as many relevant categories as possible.

Return only the categories that apply. Do not include any categories not listed.

<|im_end|>
<|im_start|>user<|im_sep|>
Process the following patient note and provide the output in the specified Python dictionary format:
Patient note:
\"\"\"{patient_note}\"\"\"
<|im_end|>
<|im_start|>assistant<|im_sep|>
"""
    
    return chat_prompt


def cat_reranking(retr, retr_txt, topk=10):
    """
    Re-rank retrieved search results from various categories.
    """
    retr_val = []
    txt_retr = []
    for res_val, res_txt in zip(retr.values(), retr_txt.values()):
        retr_val.append(res_val)
        txt_retr.append(res_txt)

    sim = {}
    sim_txt = {}
    for ls, ts in zip(retr_val, txt_retr):
        for j, (pid, simi) in enumerate(ls):
            if pid not in sim:
                sim[pid] = simi
                sim_txt[pid] = [ts[j][pid]]
            else:
                sim[pid] += simi
                sim_txt[pid].append(ts[j][pid])
    rerank_retr = dict(sorted(sim.items(), key=lambda item: item[1], reverse=True)[:topk])
    return rerank_retr, sim_txt

def commanality(input_patient, retrieved_patient):
    prompt = f"""
<|im_start|>system<|im_sep|>
You are an expert clinician and data analyst. Your task is to analyze a designated query patient note alongside a set of retrieved similar patient case summaries. For each retrieved patient note, determine the specific common relation or shared clinical observation with the input patient note. Then, provide a general relation that encapsulates the overall similarity between the input patient note and all the retrieved patient notes. Your response should list each individual relation on a separate line in the following format:

patient id <id no.> : <common relation with input patient note>

After listing all individual relations, include one final line that states the general relation between the input patient note and all the retrieved patient notes. Ensure your response is concise and avoids bullet points or numbered lists, and do not include any patient-identifying details.
<|im_end|>
<|im_start|>user<|im_sep|>
Please review the following information:

Query Patient Note:
{input_patient}

Retrieved Similar Patient Case Summaries:
{retrieved_patient}

Based on the above, for each retrieved patient note provide the common relation with the input patient note and then give a general relation between the input patient note and all the retrieved patient notes.
<|im_end|>
<|im_start|>assistant<|im_sep|>
"""
    return prompt

def main():
    # # Set up the header with the company logo positioned at the top right
    # col1, col2 = st.columns([3, 1])  # 2:1 ratio columns
    # with col1:
    #     # Use a smaller heading instead of st.title
    #     st.header("Patient Explorer")
    #     st.markdown(
    #         "###### Search through millions of clinical notes using your patient text"
    #     )

    # with col2:
    #     # Limit the image width so it doesn't get too big
    #     st.image(
    #         "/project_resources/common/retriever_v2/Miimansa logos.svg",
    #         width=5000  # adjust until it looks right
    #     )

    # col1, col2 = st.columns([3, 1])
    # with col1:
    #     # Replace st.header with a custom HTML heading tag and inline style
    #     st.markdown("<h2 style='font-size:32px;'>Patient Explorer</h2>", unsafe_allow_html=True)
    #     st.markdown("###### Search through millions of clinical notes using your patient text")
    # with col2:
    #     st.image("/project_resources/common/retriever_v2/Miimansa logos.svg", width=1000)
    
    col1, col2 = st.columns([3, 1])

    with col1:
        # Using separate markdown calls for two rows with different heading sizes
        st.markdown("<h2 style='font-size:32px;'>Patient Explorer</h2>", unsafe_allow_html=True)
        st.markdown("<h7>Search through millions of clinical notes using text description of your patient</h7>", unsafe_allow_html=True)

    with col2:
        st.image("/project_resources/common/retriever_v2/Miimansa logos.svg", use_container_width=True)


    # col1, col2 = st.columns([2, 1])

    # with col1:
    #     st.header("Patient Explorer")
    #     st.markdown("###### Search through millions of clinical notes using your patient text")

    # with col2:
    #     st.markdown(
    #         """
    #         <style>
    #         .logo-container {
    #             display: flex;
    #             align-items: center;  /* Vertical alignment */
    #             justify-content: center;  /* Horizontal alignment (optional) */
    #             height: 100%;
    #         }
    #         </style>
    #         <div class="logo-container">
    #             <img src="project_resources/common/retriever_v2/Miimansa_logos.svg" width="150">
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

    # st.markdown("---")
    # st.markdown("## Enter Patient Note")

    
    patient_note = st.text_area("Describe your patient", value="", placeholder=" ")

    if st.button("Process Note"):
        if not patient_note.strip():
            st.error("Please enter a patient note before processing.")
            return

        # Category extraction
        with st.spinner("Analyzing note..."):
            prompt_cat = construct_prompt_category(patient_note)
            response_cat = llm.invoke(prompt_cat)
            try:
                dict_str = response_cat.split('python')[-1].split('```')[0]
                dict_str = json.loads(dict_str)
                collections = {key:val['score'] for key, val in dict_str.items()}
                colln_text = {key:val['text'] for key, val in dict_str.items()}
            except Exception as e:
                st.error(f"Error parsing category response: {e}")
                return

            collections = dict(sorted(collections.items(), key=lambda item: item[1], reverse=True))
            st.write("### Categories and Confidence Scores")
            for key, score in collections.items():
                st.write(f"{key}: {score}")
                st.markdown(f"- {colln_text[key]}")

        # Search and re-ranking
        with st.spinner("Searching in collection..."):
            retr = {}
            retr_txt = {}
            for colln in collections:
                collection_name = colln.replace(' ', '_').replace('-', '_')
                # collection_name = f"{colln}_qwen".replace(' ', '_').replace('-', '_')
                try:
                    retr[colln], retr_txt[colln] = search_embeddings(query=patient_note, collection_name=collection_name, top_k=100)
                except Exception as e:
                    st.error(f"Error searching embeddings for {collection_name}: {e}")
                    continue

            rerank_retr, sim_txt = cat_reranking(retr, retr_txt, topk=10)

        # Display similar patients
        st.markdown("### Similar Patients")
        for pid, score in rerank_retr.items():
            st.markdown(f"**Patient ID:** {pid} | **Score:** {score}")
            for t in list(set(sim_txt[pid])):
                st.markdown(t)
            # Use an expander to display the full patient note without losing the main view
            with st.expander("Full Patient Note"):
                for category, notes in res[pid].items():
                    if len(notes) > 0:
                        st.markdown(f"**{category}:**")
                        for note in notes:
                            st.markdown(f"- {note}")
            st.markdown("---")

        # Commonality analysis
        # combined_text = ''
        # for pid, score in rerank_retr.items():
        #     combined_text += f'Patient id: {pid}\n'
        #     txt = "\n".join(list(set(sim_txt[pid])))
        #     combined_text += f'{txt}\n\n'
        
        # with st.spinner("Summarizing common points..."):
        #     prompt_common = commanality(patient_note, combined_text)
        #     response_common = llm.invoke(prompt_common)
        #     st.markdown("### Common Points from Patient Case Summaries")
        #     st.markdown(response_common)

if __name__ == '__main__':
    main()
