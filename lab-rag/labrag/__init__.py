"""LabRAG: ask questions about the lab's papers and get cited answers.

Successor to SHARK RAG. Papers stay where they already live (a NAS folder,
a Google Drive folder, or both); LabRAG indexes them into a single SQLite
file and answers questions with citations back to the papers.
"""

__version__ = "2.0.0"
