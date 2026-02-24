## Tutorial: Chain-of-Thought Document Analysis Exercise with Spot-Checking 

This tutorial outlines a systematic workflow for extracting, verifying, and transforming document data into various educational and technical formats using an AI collaborator.

#### Extraction & Verification
The goal here is to pull raw data and ensure the AI remains grounded in the actual text.

##### 1. Raw Triplet Extraction
**Prompt:** 
> Hi, please read the attached document. Then extract 50 entity-attribute-relationship triplets from the attached document, each representing a key sentence or phrase. Please present the triplets in the order the sentences or phrases occur in the text. Present the triplets without headings or classifications.

*Note: Do not mention the name or specific topic of the document to keep the extraction objective.*

##### 2. Spot-Check Accuracy
Verify that the triplets aren't "hallucinations" by linking them back to source sentences.
**Prompts:**
> Thanks. Now I would like to check that these triplets accurately reflect the document. Please state the sentence or phrase from the document that the first triplet represents.
> 
> Thanks. Please state the sentence or phrase from the document that the 25th triplet represents.
> 
> Thanks. Please state the sentence or phrase from the document that the last triplet represents.

---

#### Thematic Organization
Moving from a chronological list to a structured knowledge hierarchy.

##### 3. Thematic Clustering
**Prompt:**
> Thanks. Now please perform a thematic clustering of the triplets. Present the cluster labels and the triplets that represent each cluster. Print each triplet with its original sequence number. Print the total number of triplets in each cluster.

*Check: Ensure triplets link back to their original sequence number, verify cluster subtotals, and confirm a total of 50 triplets.*

##### 4. Natural Language Conversion
Transforming rigid triplets into readable study notes.
**Prompt:**
> Thanks. Now reprint the cluster labels and the triplets, but please convert the triplets into plain English bullets or short sentences. Preserve the total number of bullets and the number in each cluster from the thematic clustering. Print the original sequence number of the triplet at the end of each bullet.

*Check: Ensure bullets link back to their original sequence number, verify cluster subtotals, and confirm a total of 50 bullets.*

---

#### Exporting to Technical Formats
Using the structured data to generate presentation and web assets.

##### 5. LaTeX Beamer Slide Deck
**Prompt:**
> Thanks. Now, please generate a short LaTeX Beamer slide deck from the cluster labels and the bullets. The cluster labels should become section headers and slide titles. Please preserve the numbering of the bullets, and ensure each cluster is represented in the slide deck and that the deck contains the correct total number of bullets.

##### 6. HTML Web Page
**Prompt:**
> Thanks. Now, please generate a basic HTML page from the cluster labels and the bullets. The cluster labels should become section headers. Please preserve the numbering of the bullets, and ensure each cluster is represented in the page and that the page contains the correct total number of bullets.

*Check: Ensure bullets link back to their original sequence number, verify cluster subtotals, and confirm a total of 50 bullets.*

---

#### Assessment
Creating tools for testing knowledge and visualizing relationships.

##### 7. Question Generation
**Prompt:**
> Thanks, now please use the entity and relationship aspects of the triplets to create 50 short answer questions. Please print the questions with their cluster labels. Preserve the total number of bullets and the number in each cluster from the thematic clustering. Print the original sequence number of the triplet at the end of each question.

##### 8. Creating the Answer Key
**Prompt:**
> Thanks, now please print each question with its corresponding answer bullet using the previously created summary bullets to create question/answer pairs. Preserve the total number of questions and answers and the number of pairs in each cluster from the thematic clustering. Print the original sequence number of the triplet at the end of each answer.

*Check: Ensure Q&A pairs link back to their original sequence number, verify cluster subtotals, and confirm a total of 50 Q&A pairs.*

---

#### Interactive Visualization
Generating technical summary visuals.

##### 9. Python Word Cloud (Colab)
**Prompt:**
> Thanks, now please show me the most straightforward Python code I can use in a Colab notebook to print a word cloud of the original 50 triplets. Please include all 50 triplets as a Python list of lists in the generated example code. Include the original sequence number as a comment after each individual list. Please include any necessary !pip install statements at the beginning of the code example.

*Check: Confirm triplets in Python match original triplets, test the Python code.*

##### 10. Visual Integration (Image Handling)
**Questions:**
> What is the easiest way to display the generated word cloud image from the Colab notebook into the LaTeX generated earlier?
>
> What is the easiest way to display the generated word cloud image from the Colab notebook into the HTML generated earlier?

##### 11. Mermaid Knowledge Graph
**Prompt:**
> Thanks, now please use the first cluster of triplets to create a knowledge graph in Mermaid. Please print the text that defines the Mermaid graph, and then separately provide instructions for how to view and download it. Retain the cluster label as the graph title and sequence number of the triplets as comments in Mermaid.

*Check: Confirm Mermaid triplets match the original first cluster triplets.*
