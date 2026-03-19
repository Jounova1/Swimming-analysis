package com.swimming.repositories;

import org.springframework.data.jpa.repository.JpaRepository;

import com.swimming.models.StrokeMetric;

public interface StrokeMetricRepository extends JpaRepository <StrokeMetric,Long>
{
    
}
