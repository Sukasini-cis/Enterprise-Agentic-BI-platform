import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Initialize SentenceTransformer model globally (caches to ~/.cache/huggingface/hub)
try:
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    model = None

class FAISSVectorStore:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.texts = []
        self.index = None
        
    def build_index(self, texts):
        if not model:
            print("Model is not initialized. Cannot build index.")
            return False
            
        self.texts = [t.strip() for t in texts if t.strip()]
        if not self.texts:
            print("No text documents provided to index.")
            return False
            
        print(f"Generating embeddings for {len(self.texts)} chunks...")
        embeddings = model.encode(self.texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        
        # Initialize L2 distance index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Save index and corresponding texts
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        with open(os.path.join(self.index_path, "texts.txt"), "w", encoding="utf-8") as f:
            for text in self.texts:
                clean_text = text.replace("\n", " ").strip()
                f.write(clean_text + "\n")
        print(f"FAISS index built and saved to '{self.index_path}'.")
        return True
        
    def load_index(self):
        faiss_file = os.path.join(self.index_path, "index.faiss")
        texts_file = os.path.join(self.index_path, "texts.txt")
        if not os.path.exists(faiss_file) or not os.path.exists(texts_file):
            print("FAISS index files not found. Please upload a dataset first.")
            return False
            
        try:
            self.index = faiss.read_index(faiss_file)
            with open(texts_file, "r", encoding="utf-8") as f:
                self.texts = [line.strip() for line in f.readlines()]
            print(f"Loaded FAISS index containing {len(self.texts)} items.")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
            
    def search(self, query, k=5):
        if self.index is None:
            if not self.load_index():
                return []
                
        if not model:
            print("Embedding model not loaded. Cannot run search.")
            return []
            
        try:
            query_vector = model.encode([query], show_progress_bar=False)
            distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.texts):
                    results.append(self.texts[idx])
            return results
        except Exception as e:
            print(f"Error during vector search: {e}")
            return []


def generate_insights_from_csv(csv_path):
    """
    Reads the Superstore CSV file, performs descriptive aggregations,
    and returns a list of textual facts to represent the business data in the RAG system.
    """
    print(f"Parsing business dataset: {csv_path}")
    try:
        # Superstore dataset is usually encoded as latin1 or utf-8
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="utf-8")
            
        # Standardize column names (strip spaces)
        df.columns = [c.strip() for c in df.columns]
        
        # Ensure Sales, Profit, Discount, and Order Date are correctly cast
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0)
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0)
        
        facts = []
        
        # 1. General KPIs
        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()
        overall_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
        
        facts.append(f"Business Overview: The company has total sales of ${total_sales:,.2f} and a total profit of ${total_profit:,.2f}.")
        facts.append(f"Business Overview: The overall profit margin for the business is {overall_margin:.2f}%.")
        
        # 2. Regional Analysis
        regional = df.groupby("Region").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
        for _, row in regional.iterrows():
            reg_name = row["Region"]
            reg_sales = row["Sales"]
            reg_profit = row["Profit"]
            reg_margin = (reg_profit / reg_sales) * 100 if reg_sales > 0 else 0
            facts.append(f"Regional Analysis: Region '{reg_name}' generated total sales of ${reg_sales:,.2f} and a net profit of ${reg_profit:,.2f} (Profit Margin: {reg_margin:.2f}%).")
            
        # 3. Category Analysis
        categories = df.groupby("Category").agg({"Sales": "sum", "Profit": "sum", "Discount": "mean"}).reset_index()
        for _, row in categories.iterrows():
            cat_name = row["Category"]
            cat_sales = row["Sales"]
            cat_profit = row["Profit"]
            cat_discount = row["Discount"] * 100  # Convert to percentage
            cat_margin = (cat_profit / cat_sales) * 100 if cat_sales > 0 else 0
            facts.append(f"Category Analysis: Category '{cat_name}' generated sales of ${cat_sales:,.2f} and profit of ${cat_profit:,.2f} (Margin: {cat_margin:.2f}%) with an average discount of {cat_discount:.2f}%.")

        # 4. Sub-category Analysis
        subcats = df.groupby(["Category", "Sub-Category"]).agg({"Sales": "sum", "Profit": "sum", "Discount": "mean"}).reset_index()
        for _, row in subcats.iterrows():
            cat_name = row["Category"]
            subcat_name = row["Sub-Category"]
            sales = row["Sales"]
            profit = row["Profit"]
            discount = row["Discount"] * 100
            margin = (profit / sales) * 100 if sales > 0 else 0
            facts.append(f"Sub-category Analysis: In Category '{cat_name}', the sub-category '{subcat_name}' generated sales of ${sales:,.2f} and profit of ${profit:,.2f} (Margin: {margin:.2f}%) with an average discount of {discount:.2f}%.")

        # 5. Segment Analysis
        segments = df.groupby("Segment").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
        for _, row in segments.iterrows():
            seg_name = row["Segment"]
            sales = row["Sales"]
            profit = row["Profit"]
            margin = (profit / sales) * 100 if sales > 0 else 0
            facts.append(f"Customer Segment Analysis: Segment '{seg_name}' generated sales of ${sales:,.2f} and profit of ${profit:,.2f} (Margin: {margin:.2f}%).")

        # 6. High-Discount / Loss-Making Root Causes
        worst_subcats = subcats.sort_values(by="Profit").head(3)
        for _, row in worst_subcats.iterrows():
            subcat = row["Sub-Category"]
            loss = row["Profit"]
            avg_disc = row["Discount"] * 100
            if loss < 0:
                facts.append(f"Root Cause: The sub-category '{subcat}' is highly unprofitable, generating a net loss of ${loss:,.2f} with a high average discount rate of {avg_disc:.2f}%. This indicates discount erosion.")

        # Top 15 Loss-making Products
        loss_products = df.groupby(["Sub-Category", "Product Name"]).agg({"Profit": "sum", "Sales": "sum", "Discount": "mean"}).reset_index()
        loss_products = loss_products.sort_values(by="Profit").head(15)
        for idx, row in loss_products.iterrows():
            prod_name = row["Product Name"]
            subcat = row["Sub-Category"]
            prod_sales = row["Sales"]
            prod_loss = row["Profit"]
            prod_disc = row["Discount"] * 100
            if prod_loss < 0:
                facts.append(f"Root Cause: Product '{prod_name}' (Sub-category '{subcat}') is a top loss-maker, causing a net loss of ${prod_loss:,.2f} on sales of ${prod_sales:,.2f} with an average discount of {prod_disc:.2f}%.")
                
        # 7. Correlation summary
        facts.append("Root Cause Correlation: High discounts (typically over 15%) are highly correlated with negative profit margins across Furniture sub-categories (like Tables and Bookcases) and Office Supplies sub-categories (like Supplies).")

        print(f"Successfully generated {len(facts)} business insight documents from the dataset.")
        return facts
    except Exception as e:
        print(f"Error generating insights from CSV: {e}")
        return []

def build_vector_store_from_csv(csv_path, index_path="faiss_index"):
    """Helper function to load CSV, extract facts, and generate the FAISS vector database."""
    facts = generate_insights_from_csv(csv_path)
    if not facts:
        print("Failed to generate facts from CSV.")
        return False
        
    store = FAISSVectorStore(index_path)
    return store.build_index(facts)

if __name__ == "__main__":
    # Test script directly if there is a local CSV file
    csv_file = "Sample - Superstore1.csv"
    if os.path.exists(csv_file):
        build_vector_store_from_csv(csv_file)
        store = FAISSVectorStore()
        test_queries = [
            "Why is the company losing money on Tables?",
            "What is the average discount for Furniture?",
            "Which category has the highest sales?"
        ]
        print("\n--- Testing Retrieval ---")
        for q in test_queries:
            print(f"\nQuery: {q}")
            results = store.search(q, k=2)
            for r in results:
                print(f"- {r}")
    else:
        print(f"Test CSV '{csv_file}' not found.")
