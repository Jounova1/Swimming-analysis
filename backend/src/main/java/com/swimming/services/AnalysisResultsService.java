package com.swimming.services;
import java.util.List;

import com.swimming.models.AnalysisResults;
import com.swimming.repositories.AnalysisResultsRepository;

public class AnalysisResultsService {
    private AnalysisResultsRepository analysisResultsRepository;

    public AnalysisResultsService(AnalysisResultsRepository analysisResultsRepository) {
        this.analysisResultsRepository = analysisResultsRepository;
    }

    public List<AnalysisResults> getAllAnalysisResults() {
        return analysisResultsRepository.findAll();
    }

    public AnalysisResults getAnalysisResultsById(Long id) {
        return analysisResultsRepository.findById(id).orElse(null);
    }

    public AnalysisResults createAnalysisResults(AnalysisResults analysisResults) {
        return analysisResultsRepository.save(analysisResults);
    }

    public AnalysisResults updateAnalysisResults(Long id, AnalysisResults analysisResultsDetails) {
        AnalysisResults analysisResults = analysisResultsRepository.findById(id).orElse(null);
        if (analysisResults != null) {
            return analysisResultsRepository.save(analysisResults);
        }
        return null;
    }

    public void deleteAnalysisResults(Long id) {
        analysisResultsRepository.deleteById(id);
    }
}