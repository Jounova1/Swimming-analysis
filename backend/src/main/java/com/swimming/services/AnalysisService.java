package com.swimming.services;

import com.swimming.models.Session;
import com.swimming.models.StrokeMetric;
import com.swimming.repositories.SessionRepository;
import com.swimming.repositories.StrokeMetricRepository;
import org.springframework.stereotype.Service;

@Service
public class AnalysisService {

    private final SessionRepository sessionRepository;
    private final StrokeMetricRepository metricRepository;

    public AnalysisService(SessionRepository sessionRepository,
                           StrokeMetricRepository metricRepository) {
        this.sessionRepository = sessionRepository;
        this.metricRepository = metricRepository;
    }

    public StrokeMetric saveAnalysis(Long sessionId,double strokeRate,double velocity)
     {   
        Session session = sessionRepository.findById(sessionId)
                .orElseThrow(() -> new RuntimeException("Session not found"));

        StrokeMetric metric = new StrokeMetric();
        metric.setStrokeRate(strokeRate);
        metric.setVelocity(velocity);
        metric.setSession(session);
        return metricRepository.save(metric);
    }
}