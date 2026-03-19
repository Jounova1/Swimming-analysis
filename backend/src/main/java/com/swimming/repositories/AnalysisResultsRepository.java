package com.swimming.repositories;

import org.springframework.data.jpa.repository.JpaRepository;

import com.swimming.models.AnalysisResults;

public interface AnalysisResultsRepository extends JpaRepository <AnalysisResults,Long>
{
    
}
