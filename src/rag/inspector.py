#!/usr/bin/env python3
"""
Milvus Collection Inspector
A comprehensive tool to explore Milvus collections from console.
"""

import sys
import json
from typing import List, Dict, Any
from pymilvus import connections, Collection, utility, DataType
import argparse
from tabulate import tabulate

class MilvusInspector:
    def __init__(self, host: str = "localhost", port: str = "19530", 
                 user: str = "", password: str = "", secure: bool = False):
        """Initialize connection to Milvus."""
        self.host = host
        self.port = port
        
        try:
            # Connect to Milvus
            connections.connect(
                "default",
                host=host,
                port=port,
                user=user,
                password=password,
                secure=secure
            )
            print(f"✅ Connected to Milvus at {host}:{port}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {str(e)}")
            sys.exit(1)
    
    def list_collections(self) -> List[str]:
        """List all collections in Milvus."""
        try:
            collections = utility.list_collections()
            return collections
        except Exception as e:
            print(f"❌ Error listing collections: {str(e)}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection."""
        try:
            # Check if collection exists
            if not utility.has_collection(collection_name):
                return {"error": f"Collection '{collection_name}' does not exist"}
            
            # Get collection object
            collection = Collection(collection_name)
            
            # Get schema information
            schema = collection.schema
            fields_info = []
            
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "type": field.dtype.name,
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id,
                    "description": field.description or ""
                }
                
                # Add dimension for vector fields
                if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                    field_info["dimension"] = field.params.get("dim", "N/A")
                
                fields_info.append(field_info)
            
            # Get collection statistics
            try:
                collection.load()  # Load collection to get stats
                stats = collection.get_stats()
                row_count = stats.row_count if hasattr(stats, 'row_count') else 0
            except:
                row_count = "Unknown"
            
            # Get indexes
            indexes = []
            try:
                for field in schema.fields:
                    if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                        index_info = collection.index(field.name)
                        if index_info:
                            indexes.append({
                                "field": field.name,
                                "index_type": index_info.params.get("index_type", "Unknown"),
                                "metric_type": index_info.params.get("metric_type", "Unknown"),
                                "params": index_info.params
                            })
            except:
                indexes = []
            
            return {
                "name": collection_name,
                "description": schema.description or "",
                "fields": fields_info,
                "row_count": row_count,
                "indexes": indexes,
                "auto_id": schema.auto_id,
                "enable_dynamic_field": schema.enable_dynamic_field
            }
            
        except Exception as e:
            return {"error": f"Error getting collection info: {str(e)}"}
    
    def show_collections_summary(self):
        """Display summary of all collections."""
        collections = self.list_collections()
        
        if not collections:
            print("📝 No collections found in Milvus")
            return
        
        print(f"\n📚 Found {len(collections)} collections:")
        print("=" * 60)
        
        summary_data = []
        for collection_name in collections:
            info = self.get_collection_info(collection_name)
            
            if "error" in info:
                summary_data.append([
                    collection_name,
                    "Error",
                    "-",
                    "-",
                    info["error"][:50]
                ])
            else:
                vector_fields = [f for f in info["fields"] 
                               if f["type"] in ["FLOAT_VECTOR", "BINARY_VECTOR"]]
                vector_dims = [f.get("dimension", "N/A") for f in vector_fields]
                
                summary_data.append([
                    collection_name,
                    info["row_count"],
                    len(info["fields"]),
                    f"{len(vector_fields)} ({','.join(map(str, vector_dims))})",
                    info["description"][:30] + "..." if len(info["description"]) > 30 else info["description"]
                ])
        
        headers = ["Collection Name", "Row Count", "Fields", "Vectors (Dims)", "Description"]
        print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    
    def show_collection_details(self, collection_name: str):
        """Show detailed information about a specific collection."""
        info = self.get_collection_info(collection_name)
        
        if "error" in info:
            print(f"❌ {info['error']}")
            return
        
        print(f"\n📋 Collection Details: {collection_name}")
        print("=" * 60)
        print(f"Description: {info['description'] or 'No description'}")
        print(f"Row Count: {info['row_count']}")
        print(f"Auto ID: {info['auto_id']}")
        print(f"Dynamic Fields: {info['enable_dynamic_field']}")
        
        print(f"\n📊 Fields ({len(info['fields'])}):")
        field_data = []
        for field in info["fields"]:
            field_row = [
                field["name"],
                field["type"],
                "✅" if field["is_primary"] else "",
                "✅" if field["auto_id"] else "",
                field.get("dimension", ""),
                field["description"][:30] + "..." if len(field["description"]) > 30 else field["description"]
            ]
            field_data.append(field_row)
        
        field_headers = ["Field Name", "Type", "Primary", "Auto ID", "Dimension", "Description"]
        print(tabulate(field_data, headers=field_headers, tablefmt="grid"))
        
        if info["indexes"]:
            print(f"\n🔍 Indexes ({len(info['indexes'])}):")
            index_data = []
            for index in info["indexes"]:
                index_data.append([
                    index["field"],
                    index["index_type"],
                    index["metric_type"],
                    json.dumps(index["params"], indent=2)[:50] + "..."
                ])
            
            index_headers = ["Field", "Index Type", "Metric Type", "Parameters"]
            print(tabulate(index_data, headers=index_headers, tablefmt="grid"))
        else:
            print("\n🔍 No indexes found")
    
    def search_collections(self, pattern: str):
        """Search for collections matching a pattern."""
        collections = self.list_collections()
        matching = [c for c in collections if pattern.lower() in c.lower()]
        
        if not matching:
            print(f"🔍 No collections found matching pattern: {pattern}")
            return
        
        print(f"🔍 Collections matching '{pattern}':")
        for collection in matching:
            info = self.get_collection_info(collection)
            if "error" not in info:
                print(f"  • {collection} ({info['row_count']} rows)")
            else:
                print(f"  • {collection} (error)")
    
    def show_sample_data(self, collection_name: str, limit: int = 5):
        """Show sample data from a collection."""
        try:
            if not utility.has_collection(collection_name):
                print(f"❌ Collection '{collection_name}' does not exist")
                return
            
            collection = Collection(collection_name)
            collection.load()
            
            # Get schema to understand fields
            schema = collection.schema
            non_vector_fields = [f.name for f in schema.fields 
                               if f.dtype not in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]]
            
            if not non_vector_fields:
                print(f"⚠️  Collection '{collection_name}' only contains vector fields")
                return
            
            # Query sample data
            results = collection.query(
                expr="",  # Empty expression to get all
                output_fields=non_vector_fields[:10],  # Limit fields to avoid too much data
                limit=limit
            )
            
            if not results:
                print(f"📝 Collection '{collection_name}' is empty")
                return
            
            print(f"\n📄 Sample Data from '{collection_name}' (first {len(results)} rows):")
            print("=" * 80)
            
            # Convert to table format
            if results:
                headers = list(results[0].keys())
                table_data = []
                for row in results:
                    table_row = []
                    for header in headers:
                        value = row.get(header, "")
                        # Truncate long strings
                        if isinstance(value, str) and len(value) > 50:
                            value = value[:47] + "..."
                        table_row.append(value)
                    table_data.append(table_row)
                
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
        except Exception as e:
            print(f"❌ Error getting sample data: {str(e)}")
    
    def disconnect(self):
        """Disconnect from Milvus."""
        try:
            connections.disconnect("default")
            print("👋 Disconnected from Milvus")
        except Exception as e:
            print(f"⚠️  Error disconnecting: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Milvus Collection Inspector")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", default="19530", help="Milvus port")
    parser.add_argument("--user", default="", help="Milvus user")
    parser.add_argument("--password", default="", help="Milvus password")
    parser.add_argument("--secure", action="store_true", help="Use secure connection")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    subparsers.add_parser("list", help="List all collections")
    
    # Details command
    details_parser = subparsers.add_parser("details", help="Show collection details")
    details_parser.add_argument("collection", help="Collection name")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search collections by name")
    search_parser.add_argument("pattern", help="Search pattern")
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Show sample data")
    sample_parser.add_argument("collection", help="Collection name")
    sample_parser.add_argument("--limit", type=int, default=5, help="Number of rows to show")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = MilvusInspector(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        secure=args.secure
    )
    
    try:
        if args.command == "list" or not args.command:
            inspector.show_collections_summary()
            
        elif args.command == "details":
            inspector.show_collection_details(args.collection)
            
        elif args.command == "search":
            inspector.search_collections(args.pattern)
            
        elif args.command == "sample":
            inspector.show_sample_data(args.collection, args.limit)
            
        elif args.command == "interactive":
            # Interactive mode
            print("🔍 Interactive Milvus Inspector")
            print("Commands: list, details <name>, search <pattern>, sample <name>, quit")
            
            while True:
                try:
                    command = input("\nmilvus> ").strip().split()
                    if not command:
                        continue
                    
                    if command[0] == "quit":
                        break
                    elif command[0] == "list":
                        inspector.show_collections_summary()
                    elif command[0] == "details" and len(command) > 1:
                        inspector.show_collection_details(command[1])
                    elif command[0] == "search" and len(command) > 1:
                        inspector.search_collections(command[1])
                    elif command[0] == "sample" and len(command) > 1:
                        limit = int(command[2]) if len(command) > 2 else 5
                        inspector.show_sample_data(command[1], limit)
                    else:
                        print("Invalid command. Use: list, details <name>, search <pattern>, sample <name>, quit")
                        
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
    
    finally:
        inspector.disconnect()


if __name__ == "__main__":
    main()
