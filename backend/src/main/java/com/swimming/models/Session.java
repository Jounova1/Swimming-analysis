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
}
