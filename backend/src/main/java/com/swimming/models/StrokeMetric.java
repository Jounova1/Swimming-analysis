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

    public void setStrokeRate(double strokeRate) {
        this.stroke_rate = strokeRate;
    }

    public double getStrokeRate() {
        return stroke_rate;
    }

    public void setVelocity(double velocity) {
        this.avg_velocity = velocity;
    }

    public double getVelocity() {
        return avg_velocity;
    }

    public void setSession(Session session) {
        this.session = session;
    }

    public Session getSession() {
        return session;
    }
}
