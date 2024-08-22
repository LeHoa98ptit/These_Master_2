import streamlit as st
import json
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd


# Đọc tệp JSON
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        return json.load(file)


def display_full_wc_graph(file_path, limit=100):
    data = load_data(file_path)

    # Chỉ lấy dữ liệu đầu tiên với số lượng hạn chế
    data = data[:limit]

    # Khởi tạo đồ thị bằng networkx
    G = nx.Graph()

    # Thêm các nút và cạnh từ dữ liệu JSON
    for entry in data:
        node = entry["n"]
        relationship = entry["r"]
        match = entry["m"]

        node_label = f"{node['properties'].get('name', 'Unnamed Node')} (ID: {node['elementId']})"
        G.add_node(node["elementId"], label=node_label, type='Node')

        match_label = f"Match (ID: {match['elementId']}, Date: {match['properties'].get('date', 'N/A')})"
        G.add_node(match["elementId"], label=match_label, type='Match')

        # Thêm cạnh giữa node và match
        if node['elementId'] != match['elementId']:
            G.add_edge(node["elementId"], match["elementId"], label=relationship["type"], title=relationship["type"])

    # Tạo visualizer
    net = Network(notebook=False)

    # Thêm các nút và cạnh vào mạng lưới pyvis
    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[1]["label"], title=node[1]["label"])

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2]["label"], label=edge[2]["label"])

    # Lưu đồ thị dưới dạng file HTML
    net.write_html("wc_full.html")

    # Hiển thị đồ thị với Streamlit
    st.write("Showing detailed graph (limited to the first 100 entries):")
    st.write(
        "This graph shows a limited view of the Women's World Cup 2019 data.")
    # Đọc file HTML và hiển thị bằng Streamlit
    HtmlFile = open("wc_full.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=800)


# Hiển thị đồ thị
def display_sub_graph_wc_graph(file_path, title):
    data = load_data(file_path)

    # Khởi tạo đồ thị bằng networkx
    G = nx.Graph()

    # Thêm các nút và cạnh từ dữ liệu JSON
    for entry in data:
        person = entry["p"]
        person_label = f"{person['properties']['name']} ({', '.join(person['labels'])}, ID: {person['elementId']})"
        G.add_node(person["elementId"], label=person_label, name=person["properties"]["name"])

        for relation, related_node in zip(entry["relationships"], entry["relatedNodes"]):
            related_node_name = related_node["properties"].get("name", f"Node_{related_node['elementId']}")
            related_node_label = f"{related_node_name} ({', '.join(related_node['labels'])}, ID: {related_node['elementId']})"
            G.add_node(related_node["elementId"], label=related_node_label, name=related_node_name)

            # Thêm cạnh với tên mối quan hệ
            G.add_edge(person["elementId"], related_node["elementId"], label=relation["type"], title=relation["type"])

    # Tạo visualizer
    net = Network(notebook=False)

    # Thêm các nút và cạnh vào mạng lưới pyvis
    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[1]["label"], title=node[1]["label"])

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2]["label"], label=edge[2]["label"])

    # Lưu đồ thị dưới dạng file HTML
    net.write_html("graph.html")

    # Hiển thị đồ thị với Streamlit
    st.title(title)
    st.write(
        f"This is the visualization of the {title}. Each node represents a person or a match, and the edges represent relationships between them.")

    # Đọc file HTML và hiển thị bằng Streamlit
    HtmlFile = open("graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=800)


# Hiển thị đồ thị từ dữ liệu
def display_schema_graph(file_path, title):
    data = load_data(file_path)

    # Khởi tạo đồ thị bằng networkx
    G = nx.Graph()

    # Thêm các nút và cạnh từ dữ liệu JSON
    for entry in data:
        start_labels = entry.get("StartLabels", [])
        end_labels = entry.get("EndLabels", [])
        relationship_type = entry.get("RelationshipType", "")

        # Thêm các nút từ StartLabels
        for label in start_labels:
            if not G.has_node(label):
                G.add_node(label, label=label, type="Start")

        # Thêm các nút từ EndLabels
        for label in end_labels:
            if not G.has_node(label):
                G.add_node(label, label=label, type="End")

        # Thêm các cạnh
        for start_label in start_labels:
            for end_label in end_labels:
                G.add_edge(start_label, end_label, label=relationship_type, title=relationship_type)

    # Tạo visualizer
    net = Network(notebook=False)

    # Thêm các nút và cạnh vào mạng lưới pyvis
    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[1]["label"], title=node[1]["label"])

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2]["label"], label=edge[2]["label"])

    # Lưu đồ thị dưới dạng file HTML
    net.write_html("graph.html")

    # Hiển thị đồ thị với Streamlit
    st.title(title)
    st.write(
        f"This is the visualization of the {title}. Each node represents a label, and the edges represent relationships between them.")

    # Đọc file HTML và hiển thị bằng Streamlit
    HtmlFile = open("graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=800)


# Sinh quy tắc cho đồ thị
def generate_rules():
    st.title("Generate Rules for Graph")
    st.write("This section will allow you to generate rules for the graph.")
    # Thực hiện các thao tác cần thiết để sinh quy tắc ở đây
    # Chọn đồ thị để sinh quy tắc
    graph_option = st.selectbox(
        "Select a graph to generate rules for",
        ["Women's World Cup 2019", "CyberSecurity"]
    )

    # Chọn phương pháp
    method_option = st.selectbox(
        "Select a method",
        ["Slide Window Attention", "RAG - Retrieval Augmented Generation"]
    )

    # Chọn prompt
    prompt_option = st.selectbox(
        "Select a prompt type",
        ["Zero-shot prompt", "Few-shot prompt"]
    )

    # Chọn LLMs
    llm_option = st.selectbox(
        "Select an LLM",
        ["Llama-3", "Mixtral"]
    )

    # Hiển thị lựa chọn đã chọn
    st.write(f"Selected graph: {graph_option}")
    st.write(f"Selected method: {method_option}")
    st.write(f"Selected prompt type: {prompt_option}")
    st.write(f"Selected LLM: {llm_option}")
    # Cung cấp câu prompt cho Zero-shot
    if prompt_option == "Zero-shot prompt":
        prompt_text = (
            "Based on the following graph properties, generate detailed consistency rules "
            "(graph functional dependency and graph entity dependency). Consider the structure, "
            "node information and relationships in the graph, and provide a set of rules that can "
            "be applied to maintain consistent and accurate data.\n\n"
            "For each consistency rule you identify, provide a clear description of the rule and the "
            "corresponding Cypher query for checking."
        )
        st.write("Prompt for Zero-shot:")
        st.write(prompt_text)

    # Placeholder for Few-shot prompt (you can add details here later)
    if prompt_option == "Few-shot prompt":
        prompt_text = """
            Examples of consistency Rules:
            1. Unique Person ID: Each Person node should have a unique id.
            2. Person Node Properties: Each Person node should have a name and dob.
            3. Ensure that no two matches have the same date, stage, and tournament. This helps avoid duplicate matches within the same tournament.
            Task: Generate new rules to ensure consistency and accuracy in the graph database, considering all node types and relationships. 
            For each consistency rule you identify, provide a clear description of the rule and the corresponding Cypher query for checking

            Requirements:
            Ensure data consistency across all nodes and relationships.
            Avoid rules that apply to only one type of node or relationship broader context.
        """
        st.write("Prompt for Few-shot:")

        # Add your few-shot prompt here
        st.write(prompt_text)

    # Xử lý việc sinh quy tắc
    if st.button("Generate Rules"):
        st.write("Generating rules with the selected options...")

        if graph_option == "Women's World Cup 2019" and llm_option == "Llama-3" and prompt_option == "Zero-shot prompt" and method_option == "Slide Window Attention":
            df_rules = pd.read_csv("data/WWC_2019/wc_llama_window_zero.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "Women's World Cup 2019" and llm_option == "Mixtral" and prompt_option == "Zero-shot prompt" and method_option == "Slide Window Attention":
            df_rules = pd.read_csv("data/WWC_2019/wc_window_mixtral_zero.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "Women's World Cup 2019" and llm_option == "Llama-3" and prompt_option == "Few-shot prompt" and method_option == "Slide Window Attention":
            df_rules = pd.read_csv("data/WWC_2019/wc_window_llama_few.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "Women's World Cup 2019" and llm_option == "Mixtral" and prompt_option == "Few-shot prompt" and method_option == "Slide Window Attention":
            df_rules = pd.read_csv("data/WWC_2019/wc_window_mixtral_few.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "Women's World Cup 2019" and llm_option == "Llama-3" and prompt_option == "Zero-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/WWC_2019/wc_llama_rag_zero.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "Women's World Cup 2019" and llm_option == "Mixtral" and prompt_option == "Zero-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/WWC_2019/wc_mixtral_rag_zero.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "Women's World Cup 2019" and llm_option == "Llama-3" and prompt_option == "Few-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/WWC_2019/wc_llama_rag_few.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "Women's World Cup 2019" and llm_option == "Mixtral" and prompt_option == "Few-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/WWC_2019/wc_mixtral_rag_few.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "CyberSecurity" and llm_option == "Llama-3" and prompt_option == "Zero-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/cybersecurity/cyber_llama_rag_zero.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "CyberSecurity" and llm_option == "Mixtral" and prompt_option == "Zero-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/cybersecurity/cyber_mixtral_rag_zero.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

        if graph_option == "CyberSecurity" and llm_option == "Llama-3" and prompt_option == "Few-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/cybersecurity/cyber_llama_rag_few.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)


        if graph_option == "CyberSecurity" and llm_option == "Mixtral" and prompt_option == "Few-shot prompt" and method_option == "RAG - Retrieval Augmented Generation":
            df_rules = pd.read_csv("data/cybersecurity/cyber_mixtral_rag_few.csv")
            # Hiển thị bảng
            st.write("### Consistency Rules Table")
            st.dataframe(df_rules.style.set_properties(**{'width': '300px'}), width=1200)

    st.success("Rules have been generated successfully!")


# Sửa đồ thị
def edit_graph():
    st.title("Repair Graph")
    st.write("This section will allow you to edit the graph.")
    display_sub_graph_wc_graph('data/WWC_2019/sub_graph_wc_1.json')
    # Thực hiện các thao tác cần thiết để sửa đồ thị ở đây


# Xem thuật toán
def view_algorithm():
    st.title("View Methodologys")
    st.write("This section will display the algorithm used.")
    st.code("""
import streamlit as st
import json
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Đọc tệp JSON
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        return json.load(file)

# Hiển thị đồ thị
def display_graph(file_path, title):
    data = load_data(file_path)
    G = nx.Graph()
    for entry in data:
        person = entry["p"]
        person_label = f"{person['properties']['name']} ({', '.join(person['labels'])}, ID: {person['elementId']})"
        G.add_node(person["elementId"], label=person_label, name=person["properties"]["name"])
        for relation, related_node in zip(entry["relationships"], entry["relatedNodes"]):
            related_node_name = related_node["properties"].get("name", f"Node_{related_node['elementId']}")
            related_node_label = f"{related_node_name} ({', '.join(related_node['labels'])}, ID: {related_node['elementId']})"
            G.add_node(related_node["elementId"], label=related_node_label, name=related_node_name)
            G.add_edge(person["elementId"], related_node["elementId"], label=relation["type"], title=relation["type"])
    net = Network(notebook=False)
    for node in G.nodes(data=True):
        net.add_node(node[0], label=node[1]["label"], title=node[1]["label"])
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2]["label"], label=edge[2]["label"])
    net.write_html("graph.html")
    HtmlFile = open("graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=800)
""")


# Giao diện Streamlit
def main():
    st.sidebar.title("Menu")
    option = st.sidebar.radio("Select an option",
                              ["View Graph", "Generate Rules for Graph", "Repair Graph", "View Methodologys"])

    if option == "View Graph":
        graph_option = st.selectbox(
            "Select a graph to view",
            ["Women's World Cup 2019 Graph", "Cybersecurity Graph"]
        )

        if graph_option == "Women's World Cup 2019 Graph":
            st.write("Showing schema and detailed graph:")
            st.image("images/wc_2019.png", caption="Caption for the image", use_column_width=True)
            display_full_wc_graph('data/WWC_2019/wc_full.json')
        elif graph_option == "Cybersecurity Graph":
            st.image("images/Cyber.png", caption="Caption for the image", use_column_width=True)

    elif option == "Generate Rules for Graph":
        generate_rules()
    elif option == "Repair Graph":
        edit_graph()
    elif option == "View Methodologys":
        view_algorithm()


if __name__ == "__main__":
    main()
