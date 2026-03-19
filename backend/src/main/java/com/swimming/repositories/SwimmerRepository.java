package com.swimming.repositories;

import org.springframework.data.jpa.repository.JpaRepository;

import com.swimming.models.Swimmer;

public interface SwimmerRepository extends JpaRepository <Swimmer,Long>
{

}