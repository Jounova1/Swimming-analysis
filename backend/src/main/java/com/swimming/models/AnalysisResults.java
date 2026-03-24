package com.swimming.models;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.OneToOne;

@Entity
public class AnalysisResults {
    @Id
    @GeneratedValue

    private Long id;
    private String summary;
    
    @OneToOne
    private Session session;
}
