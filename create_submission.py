#!/usr/bin/env python3
"""
CS224V HW1 Submission Script

Creates a submission zip file containing the notebook, PDF version, and all required output files.
Performs sanity checks to ensure all expected files are present.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from typing import List, Tuple

def check_file_exists(filepath: str, description: str) -> Tuple[bool, str]:
    """Check if a file exists and return status with message."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return True, f"✅ {description}: {filepath} ({size} bytes)"
    else:
        return False, f"❌ MISSING: {description}: {filepath}"

def convert_notebook_to_pdf(notebook_path: str) -> str:
    """Convert Jupyter notebook to PDF using nbconvert."""
    pdf_path = notebook_path.replace('.ipynb', '.pdf')
    print(f"🔄 Converting notebook to PDF: {notebook_path} -> {pdf_path}")
    
    try:
        # Try using jupyter nbconvert
        result = subprocess.run([
            'jupyter', 'nbconvert', 
            '--to', 'pdf', 
            '--output', pdf_path,
            notebook_path
        ], capture_output=True, text=True, check=True)
        
        print(f"✅ Successfully converted notebook to PDF")
        return pdf_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting notebook to PDF: {e}")
        print(f"stderr: {e.stderr}")
        print("\nTrying alternative method with --no-input flag...")
        
        try:
            # Try with --no-input flag to skip problematic cells
            result = subprocess.run([
                'jupyter', 'nbconvert',
                '--to', 'pdf',
                '--no-input',
                '--output', pdf_path,
                notebook_path
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ Successfully converted notebook to PDF (no-input mode)")
            return pdf_path
            
        except subprocess.CalledProcessError as e2:
            print(f"❌ Failed to convert notebook to PDF: {e2}")
            print("Please install required dependencies:")
            print("  pip install nbconvert")
            print("  # For PDF conversion:")
            print("  # macOS: brew install --cask mactex")
            print("  # or: pip install 'nbconvert[webpdf]'")
            return None
    
    except FileNotFoundError:
        print("❌ jupyter command not found. Please install Jupyter:")
        print("  pip install jupyter nbconvert")
        return None

def main():
    print("CS224V HW1 Submission Creator")
    print("=" * 50)
    
    # Convert notebook to PDF first
    notebook_path = "notebook.ipynb"
    pdf_path = convert_notebook_to_pdf(notebook_path)
    
    # Define required files with descriptions
    required_files = [
        # Main notebook
        ("notebook.ipynb", "Main assignment notebook"),
        ("notebook.pdf", "Assignment notebook PDF (generated)") if pdf_path else None,
        
        # Action Item 3 outputs
        ("output/action_item_3_rag_response.json", "Action Item 3: Basic RAG response"),
        ("output/action_item_3_rag_response_recency.json", "Action Item 3: Recency-focused RAG response"),
        ("output/action_item_3_rag_response_recency_comment.json", "Action Item 3: Recency analysis comments"),
        ("output/action_item_3_rag_response_depth.json", "Action Item 3: Technical depth RAG response"),
        ("output/action_item_3_rag_response_depth_comment.json", "Action Item 3: Technical depth analysis comments"),
        
        # Action Item 4 outputs
        ("output/action_item_4_literature_search_response.json", "Action Item 4: Literature search response"),
        ("output/action_item_4_literature_search_response_comment.json", "Action Item 4: Literature search analysis comments"),
        
        # Action Item 5 outputs
        ("output/action_item_5_comments.json", "Action Item 5: Database exploration analysis comments"),
        
        # Action Item 6 outputs
        ("output/action_item_6_selected_theses.json", "Action Item 6: Selected theses"),
        
        # Action Item 7 outputs
        ("output/action_item_7_literature_search_response_1.json", "Action Item 7: Literature search response 1"),
        ("output/action_item_7_literature_search_response_2.json", "Action Item 7: Literature search response 2"),
        ("output/action_item_7_rag_responses_with_key_insight.json", "Action Item 7: RAG responses with key insights"),
        ("output/action_item_7_final_report_raw.md", "Action Item 7: Raw final report"),
        ("output/action_item_7_final_report.md", "Action Item 7: Final investigative report"),
        ("output/action_item_7_weaknesses_and_improvements.md", "Action Item 7: Weaknesses and improvements"),

        
    ]
    
    # Remove None entries from required_files (in case PDF conversion failed)
    required_files = [item for item in required_files if item is not None]
    
    # Check all required files
    print("\n📋 Checking required files:")
    all_required_present = True
    files_to_zip = []
    
    for filepath, description in required_files:
        exists, message = check_file_exists(filepath, description)
        print(message)
        if exists:
            files_to_zip.append(filepath)
        else:
            all_required_present = False
    
    # Final validation
    if not all_required_present:
        print("\n❌ SUBMISSION INCOMPLETE")
        print("Some required files are missing. Please complete the assignment before creating submission.")
        sys.exit(1)
    
    # Create submission zip
    submission_filename = "cs224v_hw1_submission.zip"
    print(f"\n📦 Creating submission zip: {submission_filename}")
    
    try:
        with zipfile.ZipFile(submission_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in files_to_zip:
                # Add file to zip with its relative path
                arcname = filepath
                zipf.write(filepath, arcname)
                print(f"   Added: {filepath}")
        
        # Verify the zip file was created successfully
        zip_size = os.path.getsize(submission_filename)
        print(f"\n✅ SUCCESS!")
        print(f"Submission created: {submission_filename} ({zip_size:,} bytes)")
        print(f"Total files included: {len(files_to_zip)}")
        
        # List contents for verification
        print(f"\n📋 Zip file contents:")
        with zipfile.ZipFile(submission_filename, 'r') as zipf:
            for info in zipf.infolist():
                print(f"   {info.filename} ({info.file_size:,} bytes)")
        
        print(f"\n🎉 Ready to submit: {submission_filename}")
        
    except Exception as e:
        print(f"\n❌ ERROR creating submission zip: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()