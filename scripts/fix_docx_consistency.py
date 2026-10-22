#!/usr/bin/env python3
"""
DOCX Consistency Fixer for Manuscript and Supplementary Materials.
Fixes spacing inconsistencies, reference formatting, and ensures uniformity.
"""

import re
from pathlib import Path
from docx import Document
from docx.shared import Pt
import shutil
from datetime import datetime

class DOCXConsistencyFixer:
    def __init__(self, manuscript_path, supplement_path):
        self.manuscript_path = Path(manuscript_path)
        self.supplement_path = Path(supplement_path)
        
        # Create backups
        self.backup_manuscript = self.manuscript_path.with_suffix('.docx.backup')
        self.backup_supplement = self.supplement_path.with_suffix('.docx.backup')
        
        shutil.copy(self.manuscript_path, self.backup_manuscript)
        shutil.copy(self.supplement_path, self.backup_supplement)
        
        self.manuscript = Document(self.manuscript_path)
        self.supplement = Document(self.supplement_path)
        self.fixes_applied = []
        
    def fix_millisecond_spacing(self, doc, doc_name):
        """Fix inconsistent spacing in millisecond units."""
        fixed_count = 0
        
        for para in doc.paragraphs:
            for run in para.runs:
                original_text = run.text
                
                # Pattern: number without space before ms (e.g., "0.072ms" -> "0.072 ms")
                # But preserve existing correct spacing
                text = re.sub(r'(\d)(ms)', r'\1 \2', run.text)
                
                # Pattern: number with multiple spaces before ms (e.g., "0.072  ms" -> "0.072 ms")
                text = re.sub(r'(\d)\s+(ms)', r'\1 \2', text)
                
                if text != original_text:
                    run.text = text
                    fixed_count += 1
                    self.fixes_applied.append(f"{doc_name}: Fixed millisecond spacing in paragraph")
        
        return fixed_count
    
    def fix_reference_formatting(self, doc, doc_name):
        """Ensure consistent reference formatting."""
        fixed_count = 0
        
        for para in doc.paragraphs:
            for run in para.runs:
                original_text = run.text
                
                # Fix inconsistent spacing in reference lists [1,2] -> [1, 2]
                text = re.sub(r'\[(\d+),(\d+)', r'[\1, \2', run.text)
                text = re.sub(r'\[(\d+),(\d+)', r'[\1, \2', text)  # Apply twice for multiple refs
                
                # Fix missing spaces after commas in reference ranges [1, 2,3] -> [1, 2, 3]
                text = re.sub(r',(\d)', r', \1', text)
                
                if text != original_text:
                    run.text = text
                    fixed_count += 1
                    self.fixes_applied.append(f"{doc_name}: Fixed reference formatting")
        
        return fixed_count
    
    def fix_equation_spacing(self, doc, doc_name):
        """Fix spacing around equations and mathematical expressions."""
        fixed_count = 0
        
        for para in doc.paragraphs:
            for run in para.runs:
                original_text = run.text
                
                # Fix inconsistent spacing around operators: "a+b" -> "a + b"
                # But be careful not to break existing equations
                text = re.sub(r'([a-zA-Z0-9])\+([a-zA-Z0-9])', r'\1 + \2', run.text)
                text = re.sub(r'([a-zA-Z0-9])-([a-zA-Z0-9])', r'\1 - \2', text)
                
                # Fix double spaces
                text = re.sub(r'  +', ' ', text)
                
                if text != original_text:
                    run.text = text
                    fixed_count += 1
                    self.fixes_applied.append(f"{doc_name}: Fixed equation spacing")
        
        return fixed_count
    
    def fix_unit_consistency(self, doc, doc_name):
        """Ensure consistent unit formatting."""
        fixed_count = 0
        
        for para in doc.paragraphs:
            for run in para.runs:
                original_text = run.text
                text = run.text
                
                # Standardize common units with proper spacing
                # nm (nanometers)
                text = re.sub(r'(\d)(nm)', r'\1 \2', text)
                text = re.sub(r'(\d)\s+(nm)', r'\1 \2', text)
                
                # mV (millivolts)
                text = re.sub(r'(\d)(mV)', r'\1 \2', text)
                text = re.sub(r'(\d)\s+(mV)', r'\1 \2', text)
                
                # ms (milliseconds)
                text = re.sub(r'(\d)(ms)', r'\1 \2', text)
                text = re.sub(r'(\d)\s+(ms)', r'\1 \2', text)
                
                # s (seconds) - but not in words
                text = re.sub(r'(\d)\s+(s)\b', r'\1 \2', text)
                
                # mol/m^3 formatting
                text = re.sub(r'mol\s*/\s*m\^3', 'mol m⁻³', text)
                text = re.sub(r'mol\s*m\^-3', 'mol m⁻³', text)
                
                if text != original_text:
                    run.text = text
                    fixed_count += 1
                    self.fixes_applied.append(f"{doc_name}: Fixed unit formatting")
        
        return fixed_count
    
    def fix_punctuation_spacing(self, doc, doc_name):
        """Fix punctuation spacing issues."""
        fixed_count = 0
        
        for para in doc.paragraphs:
            for run in para.runs:
                original_text = run.text
                text = run.text
                
                # Fix double spaces after periods
                text = re.sub(r'\.  +', '. ', text)
                
                # Fix missing space after periods (but not in abbreviations)
                # Be careful with this one - only fix clear cases
                text = re.sub(r'\.([A-Z])', r'. \1', text)
                
                # Fix space before punctuation
                text = re.sub(r'\s+([.,;:])', r'\1', text)
                
                # Fix multiple spaces
                text = re.sub(r'  +', ' ', text)
                
                if text != original_text:
                    run.text = text
                    fixed_count += 1
                    self.fixes_applied.append(f"{doc_name}: Fixed punctuation spacing")
        
        return fixed_count
    
    def fix_all(self):
        """Apply all consistency fixes."""
        print("Applying consistency fixes...")
        
        # Fix manuscript
        print(f"\nFixing {self.manuscript_path.name}...")
        self.fix_millisecond_spacing(self.manuscript, "Manuscript")
        self.fix_reference_formatting(self.manuscript, "Manuscript")
        self.fix_equation_spacing(self.manuscript, "Manuscript")
        self.fix_unit_consistency(self.manuscript, "Manuscript")
        self.fix_punctuation_spacing(self.manuscript, "Manuscript")
        
        # Fix supplement
        print(f"Fixing {self.supplement_path.name}...")
        self.fix_millisecond_spacing(self.supplement, "Supplement")
        self.fix_reference_formatting(self.supplement, "Supplement")
        self.fix_equation_spacing(self.supplement, "Supplement")
        self.fix_unit_consistency(self.supplement, "Supplement")
        self.fix_punctuation_spacing(self.supplement, "Supplement")
        
        # Save fixed documents
        self.manuscript.save(self.manuscript_path)
        self.supplement.save(self.supplement_path)
        
        print(f"\nFixed documents saved.")
        print(f"Backups created:")
        print(f"  {self.backup_manuscript}")
        print(f"  {self.backup_supplement}")
        
        return len(self.fixes_applied)
    
    def generate_fix_report(self, output_path="reports/docx_fixes_applied.json"):
        """Generate a report of all fixes applied."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "manuscript": str(self.manuscript_path),
            "supplement": str(self.supplement_path),
            "backup_manuscript": str(self.backup_manuscript),
            "backup_supplement": str(self.backup_supplement),
            "fixes_applied": self.fixes_applied,
            "summary": {
                "total_fixes": len(self.fixes_applied),
                "manuscript_fixes": len([f for f in self.fixes_applied if "Manuscript" in f]),
                "supplement_fixes": len([f for f in self.fixes_applied if "Supplement" in f]),
            }
        }
        
        with open(output_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        print(f"\nFix report saved to {output_path}")
        return report

def main():
    """Main function to apply consistency fixes."""
    manuscript_path = "manuscript/Manuscript_PINO_clean.docx"
    supplement_path = "manuscript/Supplementary_Materials_PINO.docx"
    
    if not Path(manuscript_path).exists():
        print(f"Error: {manuscript_path} not found")
        return
    
    if not Path(supplement_path).exists():
        print(f"Error: {supplement_path} not found")
        return
    
    fixer = DOCXConsistencyFixer(manuscript_path, supplement_path)
    num_fixes = fixer.fix_all()
    
    print(f"\n{'='*60}")
    print(f"FIXES APPLIED: {num_fixes}")
    print(f"{'='*60}")
    
    for i, fix in enumerate(fixer.fixes_applied, 1):
        print(f"{i}. {fix}")
    
    fixer.generate_fix_report()
    
    print(f"\n{'='*60}")
    print("Consistency fixes complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
