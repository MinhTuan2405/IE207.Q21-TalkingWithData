#!/bin/bash
set -e

echo "Initializing databases..."

# Create FastAPI Server user and database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create user
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$TALKWDATA_USER') THEN
            CREATE USER $TALKWDATA_USER WITH PASSWORD '$TALKWDATA_PASSWORD';
        END IF;
    END \$\$;

    -- Create database
    SELECT 'CREATE DATABASE $TALKWDATA_DB OWNER $TALKWDATA_USER'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$TALKWDATA_DB')\gexec

    -- Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE $TALKWDATA_DB TO $TALKWDATA_USER;
EOSQL

echo "Database initialized: $POSTGRES_DB (dagster), $TALKWDATA_DB (server)"
