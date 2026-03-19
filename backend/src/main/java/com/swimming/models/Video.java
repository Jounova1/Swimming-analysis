package com.swimming.models;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;

@Entity
public class Video {
    @Id
    @GeneratedValue

    private long id;
    private String filepath;

    @ManyToOne
    private Session session;
}
