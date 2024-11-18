import streamlit as st
import os
import openai
import pinecone
from openai import OpenAIError  # Updated import for error handling

# ==========================
# Set Environment Variables
# ==========================

# Replace the placeholders with your actual API keys and environment
os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "56a7767e-fcf7-427f-8cb4-6f82e59a002e"
os.environ["PINECONE_ENVIRONMENT"] = "us-east1-aws"

# ======================
# Configure OpenAI Client
# ======================
openai.api_key = os.environ["OPENAI_API_KEY"]

# =======================
# Initialize Pinecone Client
# =======================
try:
    pc=pinecone.Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# =====================
# Define Pinecone Index
# =====================
INDEX_NAME = 'perambalur-dist-search'

# Check if the index exists
if INDEX_NAME not in pc.list_indexes().names():
    st.error(f"Index '{INDEX_NAME}' does not exist in Pinecone.")
    st.stop()
else:
    index = pc.Index(INDEX_NAME)

# =====================
# Streamlit App Layout
# =====================
st.set_page_config(page_title="Perambalur District BSO Vol1 to 4 Search", layout="wide")
st.title("🔍 Perambalur District Revenue Documents Search")

# Sidebar for additional settings
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Number of Top Matches", min_value=1, max_value=20, value=10)

# Main Input
user_query = st.text_input("Enter your query:", "")

if st.button("Submit"):
    if user_query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Processing your request..."):
            try:
                # ===========================
                # Generate Embedding for Query
                # ===========================
                embedding_response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[user_query]
                )
                embedding_vector = embedding_response.data[0].embedding

                # =======================
                # Query Pinecone Index
                # =======================
                query_response = index.query(
                    vector=embedding_vector,
                    top_k=top_k,
                    include_metadata=True
                )

                matched_results = [match['metadata']['original_text'] for match in query_response['matches']]

                if not matched_results:
                    st.info("No relevant documents found.")
                else:
                    # ==============================
                    # Construct Prompt for GPT-4
                    # ==============================
                    system_prompt = (
                        "You are an assistant to Government employees in the Government of Tamil Nadu. "
                        "Provided is a set of contexts from a similarity search. Provide an accurate result as per the query:"
                    )
                    prompt = (
                        f"{system_prompt}\n\n"
                        f"**Query:** {user_query}\n\n"
                        f"**Context from similarity search:**\n" + "\n\n".join(matched_results)
                    )

                    # ==========================
                    # Get GPT-4 Response
                    # ==========================
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    )

                    final_output = response.choices[0].message.content

                    # =====================
                    # Display Results
                    # =====================
                    # First show GPT-4 Response
                    st.subheader("📝 Summarized response using GPT-4:")
                    st.write(final_output)

                    # Then show Retrieved Documents in a collapsible section
                    with st.expander("📄 View Retrieved Documents"):
                        for idx, doc in enumerate(matched_results, 1):
                            st.write(f"**Document {idx}:** {doc}")
                            st.divider()  # Add a divider between documents for better readability

            except OpenAIError as e:  # Updated error handling
                st.error(f"OpenAI API error: {e}")
            except pinecone.core.client.exceptions.PineconeException as e:
                st.error(f"Pinecone API error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
