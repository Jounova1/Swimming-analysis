package com.swimming.models;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;


@Entity
public class Session {
    @Id
    @GeneratedValue

    private long id;
    private String date;
    private String pool_length;


    @ManyToOne
    private Swimmer swimmer;

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getPool_length() {
        return pool_length;
    }

    public void setPool_length(String pool_length) {
        this.pool_length = pool_length;
    }

    public Swimmer getSwimmer() {
        return swimmer;
    }

    public void setSwimmer(Swimmer swimmer) {
        this.swimmer = swimmer;
    }
}
