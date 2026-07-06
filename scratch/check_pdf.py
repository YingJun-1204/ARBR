import os
import pypdf

def main():
    pdf_path = os.path.join("PDF", "ICLR高斯+几何注意力.pdf")
    out_path = os.path.join("scratch", "output.txt")
    
    if not os.path.exists(pdf_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("PDF file not found.\n")
        return
        
    reader = pypdf.PdfReader(pdf_path)
    
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"Total pages: {len(reader.pages)}\n\n")
        
        # We will scan all pages and write sections containing Table 2, LGA, Gaussian, or Severity
        out.write("=== Table 2 & Severity 0 Sections ===\n")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if "Table 2" in text or "Severity" in text or "severity" in text:
                out.write(f"\n--- Page {i + 1} (Table 2 / Severity) ---\n")
                out.write(text)
                out.write("\n" + "="*50 + "\n")
                
        out.write("\n=== LGA Definition & Math Sections ===\n")
        # Typically the main methodology is in pages 3-7
        for i in range(2, min(8, len(reader.pages))):
            text = reader.pages[i].extract_text()
            out.write(f"\n--- Page {i + 1} (Methodology) ---\n")
            out.write(text)
            out.write("\n" + "="*50 + "\n")

    print("Success: Written to scratch/output.txt")

if __name__ == "__main__":
    main()
