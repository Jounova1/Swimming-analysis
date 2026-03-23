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

public Long getId() {
    return id;
}

public void setId(Long id) {
    this.id = id;
}

public String getName() {
    return name;
}

public void setName(String name) {
    this.name = name;
}

public String getAge() {
    return age;
}

public void setAge(String age) {
    this.age = age;
}

public String getPhone_num() {
    return phone_num;
}

public void setPhone_num(String phone_num) {
    this.phone_num = phone_num;
}

public String getTeam() {
    return team;
}

public void setTeam(String team) {
    this.team = team;
}



}