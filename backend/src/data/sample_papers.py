"""Sample paper data for testing"""

SAMPLE_PAPERS = [
    {
        "pmid": "38123456",
        "title": "CRISPR-Cas9 Gene Editing for Treatment of Sickle Cell Disease: A Phase 1/2 Clinical Trial",
        "abstract": "Background: Sickle cell disease (SCD) is caused by a mutation in the β-globin gene. CRISPR-Cas9 gene editing offers a potential cure by reactivating fetal hemoglobin production. Methods: We conducted a phase 1/2 clinical trial involving 45 patients with severe SCD. Autologous CD34+ cells were edited using CRISPR-Cas9 to disrupt the BCL11A erythroid enhancer, followed by myeloablative conditioning and cell infusion. Results: At a median follow-up of 18 months, all patients showed sustained increases in fetal hemoglobin levels (mean 40.1%). Vaso-occlusive crises were eliminated in 93% of patients. No serious adverse events related to gene editing were observed. Conclusions: CRISPR-Cas9 gene editing shows promising efficacy and safety for treating sickle cell disease.",
        "authors": ["Zhang, Wei", "Johnson, Sarah M.", "Kim, David", "Chen, Li", "Williams, Robert"],
        "journal": "New England Journal of Medicine",
        "publication_date": "2024-03-15",
        "doi": "10.1056/NEJMoa2401234",
        "keywords": ["CRISPR-Cas9", "sickle cell disease", "gene editing", "fetal hemoglobin", "BCL11A", "clinical trial", "gene therapy"],
        "mesh_terms": ["Gene Editing", "CRISPR-Cas Systems", "Anemia, Sickle Cell", "Fetal Hemoglobin"]
    },
    {
        "pmid": "38234567",
        "title": "CAR-T Cell Therapy Targeting CD19 and CD22 for Relapsed B-Cell Acute Lymphoblastic Leukemia",
        "abstract": "Dual-targeting CAR-T cell therapy may overcome antigen escape, a major cause of relapse after single-target CAR-T treatment. We developed a bispecific CAR-T product targeting both CD19 and CD22 and evaluated its efficacy in 62 patients with relapsed/refractory B-ALL. The overall response rate was 94%, with 85% achieving complete remission with minimal residual disease negativity. At 12-month follow-up, event-free survival was 72%. Cytokine release syndrome occurred in 78% of patients (grade 3-4 in 15%). This dual-targeting approach demonstrates superior durability compared to single-target CAR-T therapy.",
        "authors": ["Park, Jennifer H.", "Liu, Xiaoming", "Thompson, Michael", "Garcia, Maria", "Lee, Sung-Ho"],
        "journal": "Nature Medicine",
        "publication_date": "2024-02-28",
        "doi": "10.1038/s41591-024-02789",
        "keywords": ["CAR-T", "CD19", "CD22", "B-ALL", "immunotherapy", "leukemia", "clinical trial", "cancer"],
        "mesh_terms": ["Receptors, Chimeric Antigen", "Precursor B-Cell Lymphoblastic Leukemia-Lymphoma", "Immunotherapy, Adoptive"]
    },
    {
        "pmid": "38345678",
        "title": "Single-Cell RNA Sequencing Reveals Tumor Microenvironment Heterogeneity in Pancreatic Ductal Adenocarcinoma",
        "abstract": "Pancreatic ductal adenocarcinoma (PDAC) has a complex tumor microenvironment that contributes to therapeutic resistance. We performed single-cell RNA sequencing on 180,000 cells from 25 PDAC tumors and matched normal tissues. We identified 15 distinct cell populations including novel cancer-associated fibroblast subtypes with immunosuppressive properties. Trajectory analysis revealed differentiation pathways of tumor-infiltrating myeloid cells toward immunosuppressive phenotypes. Spatial transcriptomics validated the co-localization of immunosuppressive cell populations. These findings provide a comprehensive atlas of PDAC heterogeneity and identify potential therapeutic targets.",
        "authors": ["Wang, Yong", "Martinez, Carlos", "Brown, Elizabeth", "Nakamura, Kenji", "Singh, Priya"],
        "journal": "Cell",
        "publication_date": "2024-01-20",
        "doi": "10.1016/j.cell.2024.01.015",
        "keywords": ["single-cell RNA-seq", "pancreatic cancer", "tumor microenvironment", "cancer-associated fibroblasts", "spatial transcriptomics", "cancer", "immunotherapy"],
        "mesh_terms": ["Single-Cell Analysis", "Carcinoma, Pancreatic Ductal", "Tumor Microenvironment", "RNA-Seq"]
    },
    {
        "pmid": "38456789",
        "title": "mRNA Vaccine Platform for Personalized Cancer Immunotherapy: First-in-Human Results",
        "abstract": "Personalized mRNA vaccines encoding tumor neoantigens represent a promising approach for cancer immunotherapy. We conducted a first-in-human trial of individualized mRNA vaccines in 30 patients with advanced solid tumors. Each vaccine encoded up to 20 patient-specific neoantigens. Neoantigen-specific T cell responses were detected in 87% of patients. Objective tumor responses were observed in 6 patients (20%), with 3 complete responses. Combination with anti-PD-1 therapy enhanced responses. The platform demonstrated feasibility, safety, and preliminary efficacy, supporting further clinical development.",
        "authors": ["Mueller, Klaus", "Anderson, Jessica", "Zhao, Ming", "O'Brien, Patrick", "Fernandez, Elena"],
        "journal": "Science",
        "publication_date": "2024-04-05",
        "doi": "10.1126/science.adh5678",
        "keywords": ["mRNA vaccine", "neoantigen", "cancer immunotherapy", "personalized medicine", "T cell response", "clinical trial", "cancer", "immunotherapy"],
        "mesh_terms": ["mRNA Vaccines", "Cancer Vaccines", "Neoantigens", "Immunotherapy"]
    },
    {
        "pmid": "38567890",
        "title": "AlphaFold3 Enables Structure-Based Drug Discovery for Previously Undruggable Protein Targets",
        "abstract": "Many disease-relevant proteins lack experimental structures, limiting structure-based drug discovery. We applied AlphaFold3 to predict structures of 500 therapeutically relevant proteins previously considered undruggable. Virtual screening against these predicted structures identified novel small molecule binders for 78 targets. Experimental validation confirmed binding for 65% of predicted hits. For the oncology target KRAS G12D, we identified a novel allosteric inhibitor with IC50 of 12 nM that showed efficacy in mouse xenograft models. This study demonstrates that AI-predicted structures can effectively guide drug discovery for challenging targets.",
        "authors": ["Patel, Ravi", "Yamamoto, Hiroshi", "Schmidt, Anna", "Robinson, Thomas", "Gupta, Neha"],
        "journal": "Nature Biotechnology",
        "publication_date": "2024-03-01",
        "doi": "10.1038/nbt.4567",
        "keywords": ["AlphaFold", "drug discovery", "protein structure", "KRAS", "virtual screening", "AI"],
        "mesh_terms": ["Drug Discovery", "Protein Conformation", "Artificial Intelligence", "Molecular Docking Simulation"]
    },
    {
        "pmid": "38678901",
        "title": "Long-term Outcomes of Base Editing for Beta-Thalassemia: 5-Year Follow-up Data",
        "abstract": "Base editing offers precise correction of disease-causing mutations without double-strand breaks. We report 5-year follow-up data from 15 patients with transfusion-dependent beta-thalassemia treated with autologous base-edited CD34+ cells. All patients achieved transfusion independence within 3 months of treatment. At 5 years, hemoglobin levels remained stable (mean 12.3 g/dL). Comprehensive genomic analyses showed no evidence of off-target editing or clonal abnormalities. Quality of life assessments demonstrated sustained improvements. These results support the long-term safety and durability of base editing for genetic diseases.",
        "authors": ["Chen, David", "Komor, Alexis", "Liu, David R.", "Green, Emily", "Santos, Ricardo"],
        "journal": "New England Journal of Medicine",
        "publication_date": "2024-05-10",
        "doi": "10.1056/NEJMoa2405678",
        "keywords": ["base editing", "beta-thalassemia", "gene therapy", "hemoglobin", "long-term follow-up", "gene editing", "clinical trial"],
        "mesh_terms": ["Gene Editing", "beta-Thalassemia", "Base Editing", "Genetic Therapy"]
    },
    {
        "pmid": "38789012",
        "title": "Gut Microbiome Signatures Predict Response to Immune Checkpoint Inhibitors in Melanoma",
        "abstract": "Response to immune checkpoint inhibitors (ICIs) varies widely among cancer patients. We analyzed gut microbiome composition in 450 melanoma patients before ICI treatment using metagenomic sequencing. Machine learning models identified microbial signatures predictive of response with 82% accuracy. High abundance of Faecalibacterium prausnitzii and Akkermansia muciniphila was associated with favorable outcomes. Fecal microbiota transplantation from responders to germ-free mice enhanced anti-tumor immunity. A microbiome-based predictive score was developed and validated in an independent cohort. These findings support microbiome profiling for treatment stratification.",
        "authors": ["Thompson, Rachel", "Belkaid, Yasmine", "Zhao, Liping", "Kumar, Vikram", "Jobin, Christian"],
        "journal": "Nature Medicine",
        "publication_date": "2024-02-15",
        "doi": "10.1038/s41591-024-02345",
        "keywords": ["gut microbiome", "immune checkpoint inhibitors", "melanoma", "biomarkers", "machine learning", "cancer", "immunotherapy"],
        "mesh_terms": ["Gastrointestinal Microbiome", "Immune Checkpoint Inhibitors", "Melanoma", "Biomarkers, Tumor"]
    },
    {
        "pmid": "38890123",
        "title": "Lipid Nanoparticle-Delivered CRISPR for In Vivo Liver Gene Editing in Humans",
        "abstract": "In vivo gene editing using lipid nanoparticle (LNP)-delivered CRISPR components offers a non-viral approach for treating genetic diseases. We conducted a phase 1 trial of LNP-CRISPR targeting PCSK9 in 12 patients with heterozygous familial hypercholesterolemia. A single intravenous infusion resulted in mean PCSK9 reduction of 65% and LDL cholesterol reduction of 52% at day 180. Editing efficiency in liver biopsies ranged from 45-67%. Transient elevations in liver enzymes were observed but resolved within 2 weeks. This study demonstrates the feasibility of in vivo LNP-CRISPR delivery for therapeutic gene editing.",
        "authors": ["Anderson, David G.", "Bhattacharya, Nandan", "Weissman, Drew", "Cullis, Pieter", "Kowalski, Philip"],
        "journal": "Nature",
        "publication_date": "2024-04-20",
        "doi": "10.1038/s41586-024-07234",
        "keywords": ["lipid nanoparticle", "CRISPR", "in vivo gene editing", "PCSK9", "liver", "hypercholesterolemia", "gene editing", "clinical trial", "gene therapy"],
        "mesh_terms": ["Nanoparticles", "CRISPR-Cas Systems", "Gene Editing", "PCSK9 Protein", "Liver"]
    },
    {
        "pmid": "38901234",
        "title": "Artificial Intelligence-Guided Antibiotic Discovery Reveals Novel Compounds Against Drug-Resistant Bacteria",
        "abstract": "The antibiotic resistance crisis demands new approaches for drug discovery. We developed a deep learning model trained on molecular structures and antibacterial activity data. Screening of 100 million virtual compounds identified 8,000 candidates with predicted activity against methicillin-resistant Staphylococcus aureus (MRSA). Experimental testing of 500 top candidates yielded 23 compounds with MIC < 2 μg/mL against MRSA. The lead compound, halicin-2, showed broad-spectrum activity including against pan-drug resistant Acinetobacter baumannii. In mouse infection models, halicin-2 demonstrated efficacy without detectable resistance development. AI-guided discovery accelerates identification of novel antibiotics.",
        "authors": ["Collins, James J.", "Stokes, Jonathan", "Yang, Kevin", "Barzilay, Regina", "Wong, Felix"],
        "journal": "Cell",
        "publication_date": "2024-03-28",
        "doi": "10.1016/j.cell.2024.02.045",
        "keywords": ["artificial intelligence", "antibiotic discovery", "MRSA", "drug resistance", "deep learning"],
        "mesh_terms": ["Drug Discovery", "Artificial Intelligence", "Anti-Bacterial Agents", "Drug Resistance, Bacterial"]
    },
    {
        "pmid": "39012345",
        "title": "Spatial Multi-Omics Analysis Reveals Mechanisms of Immunotherapy Resistance in Non-Small Cell Lung Cancer",
        "abstract": "Understanding spatial organization of the tumor immune microenvironment is crucial for overcoming immunotherapy resistance. We performed spatial multi-omics profiling (transcriptomics, proteomics, and metabolomics) on 80 non-small cell lung cancer specimens before and after anti-PD-1 therapy. Resistant tumors showed spatial exclusion of CD8+ T cells from tumor nests, associated with elevated adenosine signaling. Integration of spatial data identified tertiary lymphoid structures as hubs of anti-tumor immunity. A spatial resistance score predicted treatment outcomes with 89% accuracy. These findings reveal spatially-defined resistance mechanisms and potential combination strategies.",
        "authors": ["Lee, James", "Herbst, Roy S.", "Zhang, Feng", "Paik, Paul K.", "Rimm, David L."],
        "journal": "Cancer Cell",
        "publication_date": "2024-05-01",
        "doi": "10.1016/j.ccell.2024.04.012",
        "keywords": ["spatial multi-omics", "immunotherapy resistance", "lung cancer", "tumor microenvironment", "PD-1"],
        "mesh_terms": ["Carcinoma, Non-Small-Cell Lung", "Spatial Analysis", "Drug Resistance, Neoplasm", "Immunotherapy"]
    }
]


def search_papers(query: str, limit: int = 10, offset: int = 0, filters: dict = None):
    """Search sample papers by query"""
    query_lower = query.lower()

    results = []
    for paper in SAMPLE_PAPERS:
        # Calculate relevance score based on keyword matching
        score = 0.0

        # Check title
        if query_lower in paper["title"].lower():
            score += 0.5

        # Check abstract
        if query_lower in paper["abstract"].lower():
            score += 0.3

        # Check keywords
        for keyword in paper["keywords"]:
            if query_lower in keyword.lower():
                score += 0.2
                break

        # Check individual words
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3:
                if word in paper["title"].lower():
                    score += 0.1
                if word in paper["abstract"].lower():
                    score += 0.05

        if score > 0:
            results.append({
                **paper,
                "relevance_score": min(score, 1.0)
            })

    # Apply filters
    if filters:
        if filters.get("year_from"):
            results = [r for r in results if r.get("publication_date", "")[:4] >= str(filters["year_from"])]
        if filters.get("year_to"):
            results = [r for r in results if r.get("publication_date", "")[:4] <= str(filters["year_to"])]
        if filters.get("journals"):
            results = [r for r in results if r["journal"] in filters["journals"]]

    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Apply pagination
    total = len(results)
    results = results[offset:offset + limit]

    return total, results


def get_paper_by_pmid(pmid: str):
    """Get paper by PMID"""
    for paper in SAMPLE_PAPERS:
        if paper["pmid"] == pmid:
            return paper
    return None


def get_similar_papers(pmid: str, limit: int = 5):
    """Get similar papers based on keyword overlap"""
    target_paper = get_paper_by_pmid(pmid)
    if not target_paper:
        return []

    target_keywords = set(k.lower() for k in target_paper["keywords"])

    similar = []
    for paper in SAMPLE_PAPERS:
        if paper["pmid"] == pmid:
            continue

        paper_keywords = set(k.lower() for k in paper["keywords"])
        common = target_keywords & paper_keywords

        if common:
            similarity = len(common) / max(len(target_keywords), len(paper_keywords))
            similar.append({
                "pmid": paper["pmid"],
                "title": paper["title"],
                "similarity_score": round(similarity, 3),
                "common_keywords": list(common)
            })

    similar.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar[:limit]
