#!/usr/bin/env python3
"""
DOCX Consistency Checker and Fixer for Manuscript and Supplementary Materials.
Reviews references, formulas, terminology, and ensures consistency across documents.
"""

import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import json

class DOCXConsistencyChecker:
    def __init__(self, manuscript_path, supplement_path):
        self.manuscript_path = Path(manuscript_path)
        self.supplement_path = Path(supplement_path)
        self.manuscript = Document(manuscript_path)
        self.supplement = Document(supplement_path)
        self.issues = []
        self.fixes = []
        
    def extract_all_text(self, doc):
        """Extract all text from a document."""
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text.append(cell.text)
        return '\n'.join(text)
    
    def check_reference_format(self, doc_name, doc):
        """Check reference formatting consistency."""
        text = self.extract_all_text(doc)
        
        # Check for [1], [2, 3], [1-5] patterns
        ref_pattern = r'\[\d+(?:,\s*\d+)*(?:\s*-\s*\d+)?\]'
        refs = re.findall(ref_pattern, text)
        
        # Check for inconsistent spacing
        for ref in refs:
            if ', ' in ref and not re.match(r'\[\d+(,\s*\d+)+', ref):
                self.issues.append(f"{doc_name}: Inconsistent spacing in reference {ref}")
        
        # Check reference numbers are sequential in bibliography
        bib_refs = re.findall(r'^\[(\d+)\]', text, re.MULTILINE)
        if bib_refs:
            bib_nums = [int(r) for r in bib_refs]
            for i, num in enumerate(bib_nums[:-1]):
                if bib_nums[i+1] != num + 1:
                    self.issues.append(f"{doc_name}: Non-sequential reference number: [{num}] followed by [{bib_nums[i+1]}]")
    
    def check_mathematical_notation(self, doc_name, doc):
        """Check mathematical notation consistency."""
        text = self.extract_all_text(doc)
        
        # Check for inconsistent variable formatting
        # Variables should be italicized or in math mode
        var_pattern = r'\b([A-Za-z])_\{?(\w+)\}?\b'
        vars_found = re.findall(var_pattern, text)
        
        # Check for consistent use of Greek letters
        greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω']
        for letter in greek_letters:
            if letter in text:
                # Check if it's used consistently
                count = text.count(letter)
                if count > 0:
                    self.fixes.append(f"{doc_name}: Greek letter '{letter}' appears {count} times")
        
        # Check for inconsistent exponential notation
        exp_pattern1 = r'10\^{?(-?\d+)}?'
        exp_pattern2 = r'×\s*10\^{?(-?\d+)}?'
        exp1 = re.findall(exp_pattern1, text)
        exp2 = re.findall(exp_pattern2, text)
        
        if exp1 and exp2:
            self.issues.append(f"{doc_name}: Mixed exponential notation styles found")
    
    def check_terminology_consistency(self, doc_name, doc):
        """Check for consistent terminology."""
        text = self.extract_all_text(doc)
        
        # Check for term variations
        term_pairs = [
            ('nanopore', 'nano-pore'),
            ('DeepONet', 'Deep-O-Net', 'DeepoNet'),
            ('SG/FV', 'SGFV', 'S-G/F-V'),
            ('PDE', 'P.D.E.'),
        ]
        
        for terms in term_pairs:
            counts = [(term, text.count(term)) for term in terms if text.count(term) > 0]
            if len(counts) > 1:
                self.issues.append(f"{doc_name}: Inconsistent terminology: {counts}")
        
        # Check for consistent capitalization
        if 'pinn' in text.lower() and 'PINN' not in text:
            self.issues.append(f"{doc_name}: 'PINN' should be capitalized")
    
    def check_equation_numbering(self, doc_name, doc):
        """Check equation numbering consistency."""
        text = self.extract_all_text(doc)
        
        # Find equation references like Eq. (1), Eq. (S1), etc.
        eq_refs = re.findall(r'Eq\.\s*\((\w+)\)', text)
        
        # Check for sequential numbering
        main_eqs = [int(e) for e in eq_refs if e.isdigit()]
        supp_eqs = [e for e in eq_refs if e.startswith('S')]
        
        if main_eqs:
            for i in range(len(main_eqs) - 1):
                if main_eqs[i+1] != main_eqs[i] + 1:
                    self.issues.append(f"{doc_name}: Non-sequential equation numbering: ({main_eqs[i]}) followed by ({main_eqs[i+1]})")
    
    def check_figure_table_references(self, doc_name, doc):
        """Check figure and table references."""
        text = self.extract_all_text(doc)
        
        # Find figure references
        fig_refs = re.findall(r'Figure\s+(\d+)', text)
        fig_nums = [int(f) for f in fig_refs]
        
        # Find table references
        table_refs = re.findall(r'Table\s+(S?\d+)', text)
        
        # Check for referenced but possibly missing figures/tables
        if fig_nums:
            max_fig = max(fig_nums)
            for i in range(1, max_fig + 1):
                if i not in fig_nums:
                    self.issues.append(f"{doc_name}: Figure {i} not referenced in text")
    
    def check_units_consistency(self, doc_name, doc):
        """Check unit formatting consistency."""
        text = self.extract_all_text(doc)
        
        # Check for consistent unit formatting
        unit_patterns = [
            (r'\d+\s*nm', 'nanometers'),
            (r'\d+\s*mV', 'millivolts'),
            (r'\d+\s*ms', 'milliseconds'),
            (r'\d+\s*s\b', 'seconds'),
        ]
        
        for pattern, name in unit_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Check for inconsistent spacing
                spaced = sum(1 for m in matches if any(ch.isspace() for ch in m))
                unspaced = len(matches) - spaced
                if spaced > 0 and unspaced > 0:
                    self.issues.append(f"{doc_name}: Inconsistent spacing in {name}: {spaced} with space, {unspaced} without")
    
    def check_citation_format(self, doc_name, doc):
        """Check citation format consistency."""
        text = self.extract_all_text(doc)
        
        # Check for DOI format
        doi_pattern = r'doi:\s*10\.\d{4,}/[\w\-\.]+'
        dois = re.findall(doi_pattern, text, re.IGNORECASE)
        
        for doi in dois:
            if not re.match(r'doi:10\.\d{4,}/', doi, re.IGNORECASE):
                self.issues.append(f"{doc_name}: Malformed DOI: {doi}")
    
    def run_all_checks(self):
        """Run all consistency checks."""
        print("Running DOCX consistency checks...")
        
        # Check manuscript
        print(f"\nChecking {self.manuscript_path.name}...")
        self.check_reference_format("Manuscript", self.manuscript)
        self.check_mathematical_notation("Manuscript", self.manuscript)
        self.check_terminology_consistency("Manuscript", self.manuscript)
        self.check_equation_numbering("Manuscript", self.manuscript)
        self.check_figure_table_references("Manuscript", self.manuscript)
        self.check_units_consistency("Manuscript", self.manuscript)
        self.check_citation_format("Manuscript", self.manuscript)
        
        # Check supplement
        print(f"Checking {self.supplement_path.name}...")
        self.check_reference_format("Supplement", self.supplement)
        self.check_mathematical_notation("Supplement", self.supplement)
        self.check_terminology_consistency("Supplement", self.supplement)
        self.check_equation_numbering("Supplement", self.supplement)
        self.check_figure_table_references("Supplement", self.supplement)
        self.check_units_consistency("Supplement", self.supplement)
        self.check_citation_format("Supplement", self.supplement)
        
        return self.issues, self.fixes
    
    def generate_report(self, output_path="reports/docx_consistency_report.json"):
        """Generate a detailed consistency report."""
        report = {
            "manuscript": str(self.manuscript_path),
            "supplement": str(self.supplement_path),
            "issues": self.issues,
            "fixes": self.fixes,
            "summary": {
                "total_issues": len(self.issues),
                "total_fixes": len(self.fixes),
                "manuscript_issues": len([i for i in self.issues if "Manuscript" in i]),
                "supplement_issues": len([i for i in self.issues if "Supplement" in i]),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nConsistency report saved to {output_path}")
        return report

def main():
    """Main function to run consistency checks."""
    manuscript_path = "manuscript/Manuscript_PINO_clean.docx"
    supplement_path = "manuscript/Supplementary_Materials_PINO.docx"
    
    if not Path(manuscript_path).exists():
        print(f"Error: {manuscript_path} not found")
        return
    
    if not Path(supplement_path).exists():
        print(f"Error: {supplement_path} not found")
        return
    
    checker = DOCXConsistencyChecker(manuscript_path, supplement_path)
    issues, fixes = checker.run_all_checks()
    
    print(f"\n{'='*60}")
    print(f"CONSISTENCY CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"Issues found: {len(issues)}")
    print(f"Fixes applied: {len(fixes)}")
    
    if issues:
        print(f"\n{'='*60}")
        print("ISSUES DETECTED:")
        print(f"{'='*60}")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    if fixes:
        print(f"\n{'='*60}")
        print("FIXES APPLIED:")
        print(f"{'='*60}")
        for i, fix in enumerate(fixes, 1):
            print(f"{i}. {fix}")
    
    checker.generate_report()
    
    print(f"\n{'='*60}")
    print("Consistency check complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
