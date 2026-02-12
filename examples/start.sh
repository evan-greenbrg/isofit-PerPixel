#!bin/bash


curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d @test_call.json
