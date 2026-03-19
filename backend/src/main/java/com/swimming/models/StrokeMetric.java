package com.swimming.models;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;

@Entity
public class StrokeMetric {
    @Id
    @GeneratedValue

    private long id;
    private double stroke_rate;
    private double DPS;
    private double avg_velocity;
    private double avg_acc;
    private double kick_depth;
    private double elbow_angle;

    @ManyToOne
    private Session session;
}
