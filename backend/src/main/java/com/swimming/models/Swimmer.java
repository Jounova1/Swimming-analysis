package com.swimming.models;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;

@Entity
public class Swimmer
{
@Id
@GeneratedValue
 
private Long id;
private String name;
private String age;
private String phone_num;
private String team;

}