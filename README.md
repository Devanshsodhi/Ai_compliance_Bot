Steps to execute the scripts at the end , PLS READ THIS DOCUMENT TO UNDERSTAND WHAT I HAVE DONE


**# Ai_compliance_Bot**
Ai compliance bot for documents like invoices , purchase order and order summary etc  .
System Architecture :
![image](https://github.com/user-attachments/assets/28b932dd-f800-4342-8555-bd7baa914932)
**First Phase: Text Extraction **
The goal of this project is to automate document compliance checks using natural language queries. The system takes various types of documents—PDFs, scanned PDFs, and images—and extracts their content into structured formats for further compliance analysis. For demonstration, three computer-generated PDFs have been used:
•	Invoice
•	Purchase Order
•	Order Summary
These can be found in the input/ folder.
STEP 1 : DATA EXTRACTION AND PRE-PROCESSING : The first step is to extract text from documents, regardless of format (scanned or digitally generated).
In case of Scanned Documents: 
1.	First way how data was extracted was using Llama 3.2 Vision and Florence-2 script can be accessed through (https://www.kaggle.com/code/devanshsodhi/dataextraction) . This is OCR performed on receipts which were scanned and then data can be extracted out and converted into structured JSON using LayoutLMV models which can be fine-tuned on the type of receipts or invoices we expect of them to use further in our Ai agent. (This was not possible to implement due to shortage of time and lack of labelled data).
2.	Second way how data was extracted from scanned invoices was using tesseract OCR scripts can be accessed through (https://colab.research.google.com/drive/1ap8VLqqSYLmY4RlkYvg1Njni0UNAERsy?usp=sharing) .This extracts data using OCR but again manual parsing is required as I am unable to Finetune layout parser due to lack of data and time and While trying to use pre-trained models throws error detectron2 not detected in Layout parser . stuck to manual heuristics based and regex-based parsing in the upcoming sections. Models like Donut for automatic parsing did not yield great results as they were not being fine tuned on our data.  
Note: I Did use spacy as well and it yielded great results for scanned invoices , purchase orders and order summary pdfs but the issue was that the tokens generated were not being able to be parsed to Json or being directly sent to an LLM directly in my environment.
**In case of Computer-Generated PDFs:** need to note that in case of computer generated PDF’s we need not need to perform OCR as key-value based pairs and table data can be extracted from the pdfs in a variety of ways.
1.	Spacy & Camelot: so Spacy is primarily used for processing text as it’s a NLP library, but with the extension spacylayout it can analyse visual layout of the documents for pdfs. So, for our case spacy detects text blocks, headings, any paragraphs and it recognises tables as spans with their bounding boxes. The drawback with spacy is that it cannot exactly detect table contents and that is where Camelot comes in, Camelot helps extracting tabular data from PDFs. It uses two main functions Lattice and stream for detecting tables with and without borders explicitly. Drawback of using this method is that again data cannot be converted into usable Json as information is extracted using the bounding boxes but its not of the exact format, we expect it to be. pls access code using the following link . (https://colab.research.google.com/drive/1owLYX_SmQthBK70JWE0mJ6hpUjagxtsp?usp=sharing) .
2.	Starts with opening and processing pdf files using pdfplumber, now what this does is for each page in the pdf it extracts the text content and appends it to a string and separates the pages using some kind of predefined logic “-----x-----”  . Now using the page.extract_tables () the tables present on each page is extracted, this returns a dictionary containing all the extracted text and a list of dictionaries with list of extracted text and table data .  It is important to detect the type of document being uploaded as a pdf hence for that we use a system that detect whether it’s a invoice, purchase order or order summary based on fetching keywords and matching it to predefined search words in the program. Then the extracted texts and parsed differently depending on the detected document type using Regular Expressions and heuristics which help match the extracted data to the key value fields. The data once extracted is converted into Json and then stored as Json objects in the output folder in separate files for each type of document to be used further in the program. 
(can be viewed in the program)

3.	Layout-aware Model extraction: using LayoutLMv2 and Tesseract2 
 PDF → Images (Poppler) → OCR (Tesseract) → Formatted for LayoutLMv2 → Key-value extraction.
The issue faced with this was over and over again the model was not able to detect Detectron2 in the LayoutLMv2 and while using layout parser , and when switched to alternatives rather than using Detectron2 the models did not give good enough results . You can refer the following colab links to see my work.
. (https://colab.research.google.com/drive/1MxdYmHblELV7dc8eYxAJcGMJzEvf0hc8?usp=sharing)
(https://colab.research.google.com/drive/1J_amChX29cA-2kgIB-V7UtBOGuYlM8xc?usp=sharing)
(https://colab.research.google.com/drive/1JAhedvUvdOI0-Y1KdXyZzGRFF1OSnGw6?usp=sharing)

Conclusion : What I wanted to do was for me automate the whole process of data extraction as well and I have partially tried to apply that in the google colab notebooks before I could integrate it in my project as the methods used for automated data extraction did not yield high results as I was unable to fine tune them because of the lack of labelled data as the data labelling tools for fine tuning layoutlmv2 were paid and Google’s document ai was not working for free  and environment errors in my Laptop. Although with some help and guidance I feel I will be able to fully automate that part of the process. 

I wanted to fine tune the LILT model for automatic text extraction but again the data labelling tool used for this isnt free anymore .
https://youtu.be/EVONngnrJbE?si=Fv7TahG1XpBHLOBs

**Phase 2: LLM Integration (LangChain + Ollama) **

LangChain: for orchestrating prompts, memory, tools, and chains
Ollama: to run lightweight open-source LLM locally, LLAMA 3.1 was used in the project. 
4.	Document Parsing: from phase 1 the structured Json data for each type of document is already available and these Json objects in the output folder serve as the knowledge base for the LLM. These objects are then vector embedded and then passed to the large language model .
5.	LLM setup via Ollama: a local llm llama3.1 was used locally on the system this eliminating the need for paid APIs, models were pulled using ollama cli .
6.	Lang chain Integration: Lang chain is used as the main orchestrating framework to load the documents, convert the documents into vector embeddings suitable for the LLM , Accept users queries and then feed the queries to the model with the relevant data and return structured compliance responses or any other type of response of the query entered by the user.
Langchain is basically used for chaining together LLMs, vector stores and retrieval logic . Chroma as in-memory vector database for document retrieval. 
7.	Ollama Embeddings is used to embed JSON documents for similarity based retrieval and Vector Store Retriever is used to fetch the most relevant documents for a given query . The vector embeddings can be persisted to disk using Chroma’s persist_directory to avoid recomputation every time .
8.	Retrieval Based QA chain : LangChains retrievalQA is used to combine Document Retriever for fetching relevant Json and LLM for answering based on retrieved context .There is an interactive Questions and answer flow and IF the query contains the key word “COMPLIANCE” then additional compliance rules which are defined in natural language in the compliance_check.txt file and every time a prompt contains compliance those roles are checked and a compliance report is generated for a particular order id .

This is a demo for the CLI of the project. 
![image](https://github.com/user-attachments/assets/a8ebb5ea-27b1-4a37-aa24-dfdc16ec1c19)

To run the application on your System make sure to have Ollama installed on your computer , then download the files of this project or clone the repository 

env\Scripts\activate  

pip install -r full_requirements.txt

python main.py

python model.py 

after successfully running these commands you will be able to see a cli version of the ai agent running and asking for queries .
